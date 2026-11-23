"""Transport failures are not scientific scores. No response bodies in exceptions."""
class ProviderFailure(RuntimeError):
    def __init__(self, status):
        self.code = status
        self.failure_code = ('PROVIDER_INSUFFICIENT_BALANCE' if status == 402
                             else 'PROVIDER_EXCHANGE_FAILURE')
        super().__init__(self.failure_code)


def retryable(status):
    return status is None or status in (408, 409, 429) or (
        isinstance(status, int) and 500 <= status <= 599)


def matrix_complete(results, models=('deepseek', 'kimi', 'glm')):
    expected = {(model, task) for model in models for task in (19, 20)}
    return (len(results) == len(expected) and {(r.get('model'), r.get('task')) for r in results} == expected
            and all(r.get('status') == 'COMPLETED' and isinstance(r.get('score'), (int, float))
                    for r in results))
