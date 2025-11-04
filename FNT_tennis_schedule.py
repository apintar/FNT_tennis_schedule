import numpy as np
import pandas as pd
import sys


class FNTSchedule:
    def __init__(self, fname):
        tmp = pd.read_excel(fname, index_col=0)
        self.dates = tmp.index.to_numpy()[:-1]
        self.players = tmp.columns.to_numpy()
        self.sched = tmp.iloc[:-1, :].to_numpy()
        self.player_totals = tmp.iloc[-1, :].to_numpy()
        self.player_constraints = self.sched < 0.0
        # this can be made user defined and read from the fname file in the
        # future if needed
        self.date_totals = np.repeat(8, self.dates.shape[0])

        if self.date_totals.sum() != self.player_totals.sum():
            raise Exception("The sum of the player game totals must equal the sum of the date game totals")

        n_rows = self.sched.shape[0]
        n_cols = self.sched.shape[1]
        for i in range(n_rows):
            for j in  range(n_cols):
                if self.sched[i, j] < 0.0:
                    self.sched[i, j] = np.round(np.random.rand())


    def calc_grad(self, row, col):
        tmp = 2.0*(self.sched[row, :].sum() - self.date_totals[row])
        tmp += 2.0*(self.sched[:, col].sum() - self.player_totals[col])
        return tmp
    

    def one_update(self):
        n_changes = 0.0
        for i in np.random.permutation(self.sched.shape[0]):
            for j in np.random.permutation(self.sched.shape[1]):
                if self.player_constraints[i, j]:
                    tmp = self.calc_grad(i, j)
                    if (tmp < 0.0) & (self.sched[i, j] < 1.0):
                        self.sched[i, j] = 1.0
                        n_changes += 1.0
                    elif (tmp > 0.0) & (self.sched[i, j] > 0.0):
                        self.sched[i, j] = 0.0
                        n_changes += 1.0
        
        return n_changes


    def calc_objective(self):
        col_sums = self.sched.sum(axis=0)
        row_sums = self.sched.sum(axis=1)
        objective1 = col_sums - self.player_totals
        objective1 = objective1 * objective1
        objective2 = row_sums - self.date_totals
        objective2 = objective2 * objective2
        return objective1.sum() + objective2.sum()


    def find_schedule(self):
        while True:
            n_changes = self.one_update()
            objective = self.calc_objective()
            if objective <= 1e-8:
                # print("Success! Schedule found.")
                return 0
            if n_changes < 1.0:
                # print("Constraints not met, but no more changes to make.  See if this scheduel meets your needs.  Otherwise, try again starting from a new random schedule.")
                return 1
    

    def sched_to_df(self):
        final_sched = pd.DataFrame(self.sched)
        final_sched.columns = self.players
        final_sched.loc[len(final_sched.index)] = self.sched.sum(axis=0)
        final_sched.index = np.append(np.datetime_as_string(self.dates, unit="D"), [""])
        final_sched["Date Totals"] = np.append(self.sched.sum(axis=1), self.sched.sum())
        return final_sched



if __name__ == "__main__":

    n_iter = 0
    while True:
        sched = FNTSchedule(sys.argv[1])
        return_code = sched.find_schedule()
        if return_code == 0:
            break
        else:
            n_iter += 1

        if n_iter > int(sys.argv[2]):
            print("Could not find a solution after "+str(n_iter)+" iterations!  Quitting.")
            break

    final_sched = sched.sched_to_df()
    final_sched.to_excel(sys.argv[1].split(".")[0]+"_final.xlsx")
