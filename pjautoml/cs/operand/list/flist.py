from pjautoml.cs.css import CSS
from pjml.util.distributions import choice


class ListCS(CSS):
    """Finite Config Space (FCS) is a representation of a discrete CS.
    """

    def __init__(self, *css):
        super().__init__(*css)

        self.current_index = -1
        self.size = len(self.css)

    def __iter__(self):
        return self.css.__iter__()

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.size:
            self.current_index = -1
            raise StopIteration('No more objects.')
        return self.css[self.current_index]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.css[idx]


class CList(ListCS):  # Circular List
    def sample(self):
        """The sample method will behave like a circular list.
        Therefore, it returns the first example to the last and the first again.
        """
        if self.current_index >= self.size:
            self.current_index = -1
        self.current_index += 1
        return self.css[self.current_index]


class FList(ListCS):  # List
    def sample(self):
        """The sample method will behave like a circular list.
        Therefore, it returns the first example to the last and the first again.
        """
        return choice(self.css)
