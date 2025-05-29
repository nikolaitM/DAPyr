
class MismatchModelSize(Exception):
      def __init__(self, Nx1, Nx2):
            self.message = 'Experiment Models do not match. {} != {}'.format(Nx1, Nx2)
            super().__init__(self.message)

class MismatchTimeSteps(Exception):
      def __init__(self, T1, T2):
            self.message = 'Experiment timesteps do not match. {} != {}'.format(T1, T2)
            super().__init__(self.message)

class MismatchObs(Exception):
      def __init__(self, obf1, obf2):
            self.message = 'Experiment Obs Frequencies do not match. {} != {}'.format(obf1, obf2)
            super().__init__(self.message)

class MisMatchEnsSize(Exception):
      def __init__(self, Ne1, Ne2):
            self.message = 'Experiment Ensemble Size greater than Experiment copying from. {} > {}'.format(Ne1, Ne2)
            super().__init__(self.message)
