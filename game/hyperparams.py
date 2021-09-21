import os


class Hp:
    def __init__(self,
                 hand_size=5,
                 nlab1=5,
                 nlab2=5,
                 shuffle_cards=False,
                 agent_type='Att3',
                 opt='adam',
                 nepisodes=10000,
                 batch_size=512,
                 eps_scheme={'eps_start': 0.95, 'eps_end': 0.05, 'eps_decay': 25000},
                 replay_capacity=25000,
                 update_frequency=1,
                 ):
        self.nlab1 = nlab1  # label 1 can be number in Hanabi
        self.nlab2 = nlab2  # label 2 can be color in Hanabi
        self.hand_size = hand_size  # the number of cards held by a player
        self.shuffle_cards = shuffle_cards
        self.label1_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        self.label2_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

        self.agent_type = agent_type
        self.nepisodes = nepisodes
        self.batch_size = batch_size
        self.eps_scheme = eps_scheme
        self.replay_capacity = replay_capacity
        self.update_frequency = update_frequency
        self.opt = opt
        self.lr_sgd = 0.5
        self.lr_adam = 0.001

    def __str__(self):
        return str(self.agent_type) + '_hand_' + str(self.hand_size) + '_l1_' + str(self.nlab1) + '_l2_' + str(
            self.nlab2)

    def log(self, res_path='', file_name='hp.txt'):
        with open(os.path.join(res_path, file_name), "w") as file:
            file.write(f"agent_type {self.agent_type} \n")
            file.write(f"nlab 1 {self.nlab1} \n")
            file.write(f"nlab 2 {self.nlab2} \n")
            file.write(f"hand_size {self.hand_size} \n")
            file.write(f"shuffle_cards {self.shuffle_cards} \n")
            file.write(f"label1_list {self.label1_list} \n")
            file.write(f"label2_list {self.label2_list} \n")
            file.write(f"nepisodes {self.nepisodes} \n")
            file.write(f"batch_size {self.batch_size} \n")
            file.write(f"eps_scheme {self.eps_scheme} \n")
            file.write(f"replay_capacity {self.replay_capacity} \n")
            file.write(f"update_frequency {self.update_frequency} \n")
            file.write(f"opt {self.opt} \n")
            file.close()


hp_default = Hp()
