import warnings

import matplotlib
from sklearn.metrics import f1_score
from tqdm import tqdm

matplotlib.use("agg")

# Import TeLL
from TeLL.config import Config
from TeLL.regularization import regularize
from TeLL.session import TeLLSession
from TeLL.utility.misc import AbortRun
from TeLL.utility.timer import Timer

# Import Tensorflow
if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import torch
    from pyll_functions import invoke_dataset_from_config


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def validate(val_loader, n_classes, session, loss, prediction, model, workspace, step, batchsize, tell):
    # VALIDATION
    n_mbs = len(val_loader)
    val_batchsize = val_loader.batch_size
    loss_sum = 0
    val_predictions = np.zeros(shape=(n_mbs * val_batchsize, n_classes))
    val_labels = np.zeros(shape=(n_mbs * val_batchsize, n_classes))
    
    for vmbi, vmb in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
        val_loss, val_pred = session.run([loss, prediction],
                                         feed_dict={model.X: vmb['input'].squeeze().numpy(),
                                                    model.y_: vmb['target'].numpy(),
                                                    model.dropout: 0})
        loss_sum += val_loss
        
        val_predictions[vmbi * len(vmb['ID']):(vmbi + 1) * len(vmb['ID']), :] = val_pred
        val_labels[vmbi * len(vmb['ID']):(vmbi + 1) * len(vmb['ID']), :] = vmb['target'].numpy()
    
    np.savetxt("{}/{}_samples-predictions.txt".format(workspace.result_dir, step * batchsize),
               val_predictions, delimiter=',')
    np.savetxt("{}/{}_samples-labels.txt".format(workspace.result_dir, step * batchsize),
               val_labels, delimiter=',')
    
    # F1 score
    class_f1s = []
    for i in range(n_classes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_f1 = f1_score(y_true=val_labels[:, i].astype(int),
                                y_pred=(val_predictions[:, i] + 0.5).astype(int))
        class_f1s.append(class_f1)
    avg_f1 = np.mean(class_f1s)
    
    print(' * Validation F1-Score {f1:.3f}'.format(f1=avg_f1))
    tell.save_checkpoint(global_step=step)
    
    return avg_f1


def update_step(loss, config, tell, lr, trainables):
    # GRADIENTS
    gradients = tf.gradients(loss, trainables)
    gradient_names_before_clip = [g.op.inputs[-1].name.split("/")[1] + "_bias" if "Reshape" in g.op.inputs[-1].name else
                                  g.op.inputs[-1].name.split("/")[1] for g in gradients]
    # Gradient clipping
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    gradient_name_dict = dict(zip([g.op.inputs[-1].name for g in gradients], gradient_names_before_clip))
    
    # OPTIMIZER
    if config.get_value("lrs", False):
        momentum = config.get_value("optimizer_params")["momentum"]
        update = tf.train.MomentumOptimizer(lr, momentum).apply_gradients(zip(gradients, trainables))
    else:
        update = tell.tf_optimizer.apply_gradients(zip(gradients, trainables))
    return update, gradients, gradient_name_dict


def main(_):
    config = Config()
    np.random.seed(config.get_value("random_seed", 12345))
    
    # PARAMETERS
    n_epochs = config.get_value("epochs", 100)
    batchsize = config.get_value("batchsize", 8)
    n_classes = config.get_value("n_classes", 13)
    dropout = config.get_value("dropout", 0.25)  # TODO
    num_threads = config.get_value("num_threads", 5)
    initial_val = config.get_value("initial_val", True)
    
    # READER, LOADER
    readers = invoke_dataset_from_config(config)
    reader_train = readers["train"]
    reader_val = readers["val"]
    train_loader = torch.utils.data.DataLoader(reader_train, batch_size=config.batchsize, shuffle=True,
                                               num_workers=num_threads)
    val_loader = torch.utils.data.DataLoader(reader_val, batch_size=1, shuffle=False, num_workers=num_threads)
    
    # CONFIG
    tell = TeLLSession(config=config, model_params={"shape": reader_train.shape})
    # Get some members from the session for easier usage
    session = tell.tf_session
    model = tell.model
    workspace, config = tell.workspace, tell.config
    
    prediction = tf.sigmoid(model.output)
    prediction_val = tf.reduce_mean(tf.sigmoid(model.output), axis=0, keepdims=True)
    
    # LOSS
    if hasattr(model, "loss"):
        loss = model.loss()
    else:
        with tf.name_scope("Loss_per_Class"):
            loss = 0
            for i in range(n_classes):
                loss_batch = tf.nn.sigmoid_cross_entropy_with_logits(logits=model.output[:, i], labels=model.y_[:, i])
                loss_mean = tf.reduce_mean(loss_batch)
                loss += loss_mean
    
    # Validation loss after patching
    if hasattr(model, "loss"):
        loss_val = model.loss()
    else:
        with tf.name_scope("Loss_per_Class_Patching"):
            loss_val = 0
            for i in range(n_classes):
                loss_batch = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reduce_mean(model.output[:, i], axis=0, keepdims=True), labels=model.y_[:, i])
                loss_mean = tf.reduce_mean(loss_batch)
                loss_val += loss_mean
    
    # REGULARIZATION
    reg_penalty = regularize(layers=model.layers, l1=config.l1, l2=config.l2, regularize_weights=True,
                             regularize_biases=True)
    
    # LEARNING RATE (SCHEDULE)
    # if a LRS is defined always use MomentumOptimizer and pass learning rate to optimizer
    lrs_plateu = False
    if config.get_value("lrs", None) is not None:
        lr_sched_type = config.lrs["type"]
        if lr_sched_type == "plateau":
            lrs_plateu = True
            learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            lrs_learning_rate = config.get_value("optimizer_params")["learning_rate"]
            lrs_n_bad_epochs = 0  # counter for plateu LRS
            lrs_patience = config.lrs["patience"]
            lrs_factor = config.lrs["factor"]
            lrs_threshold = config.lrs["threshold"]
            lrs_mode = config.lrs["mode"]
            lrs_best = -np.inf if lrs_mode == "max" else np.inf
            lrs_is_better = lambda old, new: (new > old * (1 + lrs_threshold)) if lrs_mode == "max" else (
                    new < old * (1 - lrs_threshold))
    else:
        learning_rate = None  # if no LRS is defined the default optimizer is used with its defined learning rate
    
    # LOAD WEIGHTS and get list of trainables if specified
    assign_loaded_variables = None
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if config.get_value("checkpoint", None) is not None:
        with Timer(name="Loading Checkpoint", verbose=True):
            assign_loaded_variables, trainables = tell.load_weights(config.get_value("checkpoint", None),
                                                                    config.get_value("freeze", False),
                                                                    config.get_value("exclude_weights", None),
                                                                    config.get_value("exclude_freeze", None))
    
    # Update step
    if len(trainables) > 0:
        update, gradients, gradient_name_dict = update_step(loss + reg_penalty, config, tell, lr=learning_rate,
                                                            trainables=trainables)
    
    # INITIALIZE Tensorflow VARIABLES
    step = tell.initialize_tf_variables().global_step
    
    # ASSING LOADED WEIGHTS (overriding initializations) if available
    if assign_loaded_variables is not None:
        session.run(assign_loaded_variables)
    
    # -------------------------------------------------------------------------
    # Start training
    # -------------------------------------------------------------------------
    try:
        n_mbs = len(train_loader)
        epoch = int((step * batchsize) / (n_mbs * batchsize))
        epochs = range(epoch, n_epochs)
        
        if len(trainables) == 0:
            validate(val_loader, n_classes, session, loss_val, prediction_val, model,
                     workspace, step, batchsize, tell)
            return
        
        print("Epoch: {}/{} (step: {}, nmbs: {}, batchsize: {})".format(epoch + 1, n_epochs, step, n_mbs, batchsize))
        for ep in epochs:
            if ep == 0 and initial_val:
                f1 = validate(val_loader, n_classes, session, loss_val, prediction_val, model,
                              workspace, step, batchsize, tell)
            else:
                if config.has_value("lrs_best") and config.has_value("lrs_learning_rate") and config.has_value(
                        "lrs_n_bad_epochs"):
                    f1 = config.get_value("lrs_f1")
                    lrs_best = config.get_value("lrs_best")
                    lrs_learning_rate = config.get_value("lrs_learning_rate")
                    lrs_n_bad_epochs = config.get_value("lrs_n_bad_epochs")
                else:
                    f1 = 0
            
            # LRS "Plateu"
            if lrs_plateu:
                # update scheduler
                if lrs_is_better(lrs_best, f1):
                    lrs_best = f1
                    lrs_n_bad_epochs = 0
                else:
                    lrs_n_bad_epochs += 1
                # update learning rate
                if lrs_n_bad_epochs > lrs_patience:
                    lrs_learning_rate = max(lrs_learning_rate * lrs_factor, 0)
                    lrs_n_bad_epochs = 0
            
            with tqdm(total=len(train_loader), desc="Training [{}/{}]".format(ep + 1, len(epochs))) as pbar:
                for mbi, mb in enumerate(train_loader):
                    # LRS "Plateu"
                    if lrs_plateu:
                        feed_dict = {model.X: mb['input'].numpy(), model.y_: mb['target'].numpy(),
                                     model.dropout: dropout,
                                     learning_rate: lrs_learning_rate}
                    else:
                        feed_dict = {model.X: mb['input'].numpy(), model.y_: mb['target'].numpy(),
                                     model.dropout: dropout}
                    
                    # TRAINING
                    pred, loss_train, _ = session.run([prediction, loss, update], feed_dict=feed_dict)
                    
                    # Update status
                    pbar.set_description_str("Training [{}/{}] Loss: {:.4f}".format(ep + 1, len(epochs), loss_train))
                    pbar.update()
                    step += 1
            
            validate(val_loader, n_classes, session, loss_val, prediction_val, model,
                     workspace, step, batchsize, tell)
    except AbortRun:
        print("Aborting...")
    finally:
        tell.close(global_step=step, save_checkpoint=True)


if __name__ == "__main__":
    tf.app.run()
