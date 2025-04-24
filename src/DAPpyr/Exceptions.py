
class MismatchModelSize(Exception):
      def __init__(self, Nx1, Nx2):
            self.message = 'Experiment Models do not match. {} != {}'.format(Nx1, Nx2)
            super().__init__(self.message)

class MismatchTimeSteps(Exception):
      def __init__(self, T1, T2):
            self.message = 'Experiment timesteps do not match. {} != {}'.format(T1, T2)
            super().__init__(self.message)

class MismatchObs(Exception):
      def __init__(self, Ny1, Ny2):
            self.message = 'Experiment Number of Obs do not match. {} != {}'.format(Ny1, Ny2)
            super().__init__(self.message)
