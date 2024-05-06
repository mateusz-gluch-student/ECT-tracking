import os
from numpy import ndarray
from dataclasses import dataclass

from .filepath import FilepathLoader

@dataclass
class SequenceLoader(FilepathLoader):

    dirpath: str | None = None
    root_name: str | None = None
    ftype: str | None = None
    index: int = 1
    filepath: str | None = None
    image: ndarray | None = None

    def _iterate(self):
        self.index += 1
        self.filepath = f"{self.dirpath}/{self.root_name}{self.index:02d}.{self.ftype}"


    def load(self) -> ndarray:
        self._iterate()
        return super().load()


    def load_batch(self) -> ndarray:
        self._iterate()

        while os.path.isfile(self.filepath):    
            print(f"Loading {self.filepath}")
            yield super().load(self.filepath)
            self._iterate()
        