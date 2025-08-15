import os
import sys
import contextlib
import io


@contextlib.contextmanager
def suppress_omp_nested_warning():
    try:
        # This works only outside of Jupyter
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()

        with open(os.devnull, 'w') as devnull:
            devnull_fd = devnull.fileno()

            # Save original fds
            saved_stdout = os.dup(original_stdout_fd)
            saved_stderr = os.dup(original_stderr_fd)

            try:
                os.dup2(devnull_fd, original_stdout_fd)
                os.dup2(devnull_fd, original_stderr_fd)
                yield
            finally:
                os.dup2(saved_stdout, original_stdout_fd)
                os.dup2(saved_stderr, original_stderr_fd)
                os.close(saved_stdout)
                os.close(saved_stderr)

    except (io.UnsupportedOperation, AttributeError):
        # Fallback for Jupyter: suppress Python-level output only
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
