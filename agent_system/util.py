import sys
import os


class HiddenPrints:
    hide_prints = False

    def __init__(self, model_name=None, console=None, use_newline=True):
        self.model_name = model_name
        self.console = console
        self.use_newline = use_newline
        self.tqdm_aux = None

    def __enter__(self):
        if self.hide_prints:
            import tqdm  # We need to do an extra step to hide tqdm outputs. Does not work in Jupyter Notebooks.

            def nop(it, *a, **k):
                return it

            self.tqdm_aux = tqdm.tqdm
            tqdm.tqdm = nop

            if self.model_name is not None:
                self.console.print(f'Loading {self.model_name}...')
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            # May not be what we always want, but some annoying warnings end up to stderr
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide_prints:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            sys.stdout = self._original_stderr
            if self.model_name is not None:
                self.console.print(f'{self.model_name} loaded ')
            import tqdm
            tqdm.tqdm = self.tqdm_aux
