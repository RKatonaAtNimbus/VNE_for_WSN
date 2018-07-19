'''
Calculate the weight of a link that is used for the shortest path calculations as a composite metric.
'''

class LinkCost():

    '''Initialize link metric attributes'''
    def __init__(self, plr, load):
        self.__attr_weights = []  # weight values used to control the influence of the attributes used for the link metric
        self.__link_weight = 0  # composite value calculated from the below indicators
        self.__plr = plr
        self.__load = load
        self.__attr_weights = [40, 3, 2, 1]
        self.__RBW = 100000
        self.__bandwidth = 128
        self.__delay = 50
        #self.__bandwidth = bandwidth
        #self.__delay = delay

    '''Setters/getters for fields'''
    @classmethod
    def set_RBW(self, rbw):
        self.__RBW = rbw

    @classmethod
    def get_RBW(self):
        return self.__RBW

    @classmethod
    def set_bandwidth(self, bw):
        self.__bandwidth = bw

    @classmethod
    def get_bandwidth(self):
        return self.__bandwidth

    @classmethod
    def set_delay(self, delay):
        self.__delay = delay

    @classmethod
    def get_delay(self):
        return self.__delay

    @classmethod
    def set_att_weights(self, w):
        self.__attr_weights = w

    @classmethod
    def set_att_weights(self):
        return self.__attr_weights

    '''Return the weight associated  with the link'''
    def get_weight(self, link):
      return self.calculate_weight(link)

    '''Calculate link weight based on composite metric formula'''
    def calculate_weight(self, link):

        if(link.__bandwidth <= 0):
            link.__bandwidth = 1
        if (link.__delay <= 0):
            link.__delay = 1
        if (link.__plr <= 0):
            link.__plr = 1
        if (link.__load <= 0):
            link.__load = 1

        self.__link_weight = (self.__attr_weights[0] * link.__load)+(self.__attr_weights[1] * link.__delay)+(self.__attr_weights[2] * link.__plr)+(self.__attr_weights[3] * (self.__RBW//link.__bandwidth))
        return self.__link_weight













