class TerminalCheckerHelper:

    # Sometimes we want to calculate tail of the next state (Unext, Qnext...) if the episode has finished
    # because of time limit
    @staticmethod
    def truncated_std_checker(truncated, props):
        try:
            result = props['TimeLimit.truncated']
        except KeyError:
            result = False
        return not (truncated or result)
