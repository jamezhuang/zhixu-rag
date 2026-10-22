from contextvars import ContextVar


class GenerationCancelled(Exception):
    pass


cancel_check = ContextVar('cancel_check', default=None)


def check_cancelled():
    checker = cancel_check.get()
    if checker:
        checker()


def cancellable_node(fn):
    def invoke(state):
        check_cancelled()
        result = fn(state)
        check_cancelled()
        return result
    return invoke
