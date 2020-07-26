from pjautoml.cs.cs import CS


class CSS(CS):
    def __init__(self, *css):
        # This for-comprehension is done to guarantee that everything received
        # is a config space.
        # You could pass a Component, it will be transformed into a Singleton
        # CS by this for.
        css = [cs.cs for cs in css]

        super().__init__({"css": css})
        self.css = css
