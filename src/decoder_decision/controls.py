"""Pure global-step schedule and dev-only checkpoint selection controls."""
import math


def learning_rate(step, peak=1e-5, floor=1e-6, warmup=20, decay=814):
    assert step >= 1 and decay > warmup > 0 and peak >= floor > 0
    if step <= warmup:
        return peak * step / warmup
    t = min(1., (step - warmup) / (decay - warmup))
    return floor + (peak - floor) * (1 + math.cos(math.pi * t)) / 2


def evaluation_due(step, steps_per_epoch, interval=200, merge=50):
    position = step % steps_per_epoch
    return position == 0 or (step % interval == 0 and
                            merge < position < steps_per_epoch - merge)


def update_selection(state, loss, step, first_epoch=814, delta=.005, patience=3):
    """Track absolute best separately from significant improvement/patience.

    Patience starts at the first completed epoch; earlier evaluations establish
    its improvement reference but cannot consume the post-epoch patience.
    """
    assert math.isfinite(loss)
    state = dict(state)
    improved = loss < state.get('best_loss', float('inf'))
    if improved:
        state.update(best_loss=loss, best_step=step)
    significant = loss <= state.get('significant_loss', float('inf')) - delta
    if significant:
        state.update(significant_loss=loss, bad_windows=0)
    elif step >= first_epoch:
        state['bad_windows'] = state.get('bad_windows', 0) + 1
    if step < first_epoch:
        state['bad_windows'] = 0
    state['stop'] = step >= first_epoch and state.get('bad_windows', 0) >= patience
    return state, improved
