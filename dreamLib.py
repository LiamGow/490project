from myLib import K

fetch_loss_and_grads = None

def set_FLGRAD(loss, dream):
    global fetch_loss_and_grads

    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # normalize
    outputs = [loss, grads]

    fetch_loss_and_grads = K.function([dream], outputs)

# deep dream functions
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        x += step * grad_values
    return x
