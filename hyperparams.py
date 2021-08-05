class Hp:
    def __init__(self,
                 hand_size=5,
                 nlab1=5,
                 nlab2=5,
                 shuffle_cards=False,
                 opt='adam',
                 nepsidoes=1000000,
                 batch_size=512,
                 eps_scheme={'eps_start': 0.95, 'eps_end': 0.05, 'eps_decay': 25000},
                 replay_capacity=25000,
                 update_frequency=100,
                 ):
        self.nlab1 = nlab1  # label 1 can be number in Hanabi
        self.nlab2 = nlab2  # label 2 can be color in Hanabi
        self.hand_size = hand_size  # the number of cards held by a player
        self.shuffle_cards = shuffle_cards
        self.label1_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        self.label2_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

        self.nepisodes = nepsidoes
        self.batch_size = batch_size
        self.eps_scheme = eps_scheme
        self.replay_capacity = replay_capacity
        self.update_frequency = update_frequency
        self.opt = opt
        self.lr_sgd = 0.5
        self.lr_adam = 0.001

    def __str__(self):
        return 'hand_' + str(self.hand_size) + '_l1_' + str(self.nlab1) + '_l2_' + str(self.nlab2)


hp_default = Hp()
