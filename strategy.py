from enum import Enum


class AdaptStrategy(Enum):
    LeftTop = 0
    LeftBottom = 1
    RightTop = 2
    RightBottom = 3
    Center = 4
    Rand = 5

    @staticmethod
    def change_strategy(instance, adapt_strategy, cols=5, rows=5):
        new_strategy = adapt_strategy
        if instance.group == 'g0':
            if instance.text == 'left':
                if adapt_strategy.value == 2: new_strategy = AdaptStrategy.LeftTop
                elif adapt_strategy.value == 3: new_strategy = AdaptStrategy.LeftBottom
            else:
                if adapt_strategy.value == 0: new_strategy = AdaptStrategy.RightTop
                elif adapt_strategy.value == 1: new_strategy = AdaptStrategy.RightBottom
        elif instance.group == 'g1':
            if instance.text == 'top':
                if adapt_strategy.value == 1: new_strategy = AdaptStrategy.LeftTop
                elif adapt_strategy.value == 3: new_strategy = AdaptStrategy.RightTop
            else:
                if adapt_strategy.value == 0: new_strategy = AdaptStrategy.LeftBottom
                elif adapt_strategy.value == 2: new_strategy = AdaptStrategy.RightBottom
        elif instance.group == 'g2':
            if instance.state == "down":
                new_strategy = AdaptStrategy.Rand if instance.text == 'rand' else AdaptStrategy.Center
            else: new_strategy = AdaptStrategy.RightBottom

        if new_strategy.value in [0, 1]: top_col = cols - 1
        elif new_strategy.value in [2, 3]: top_col = 0
        else: top_col = int(cols * .5)

        if new_strategy.value in [1, 3]: top_row = 0
        elif new_strategy.value in [0, 2]: top_row = rows - 1
        else: top_row = int(rows * .5)

        return new_strategy, top_col, top_row
