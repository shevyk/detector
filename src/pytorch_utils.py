class EarlyStopping():
    def __init__(self, patience, best_result):
        self.patience = patience
        self.no_improvement = 0
        self.valid_best_result = best_result
    
    def stop(self, train_loss, valid_loss, epoch):
        if valid_loss < self.valid_best_result:
            # new loss is an improvement on the best result so far
            self.valid_best_result = valid_loss
            self.train_best_result = train_loss
            self.epoch = epoch
            # restart count of iterations without improvement 
            self.no_improvement = 0