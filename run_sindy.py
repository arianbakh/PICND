import numpy as np
import sys

from iran_stock import get_iran_stock_networks


# Algorithm Settings
SINDY_ITERATIONS = 10
CANDIDATE_LAMBDAS = [2 ** -i for i in range(101)]


def _get_theta(x, i):
    time_frames = x.shape[0]
    x_i = x[:time_frames, i]
    column_list = [np.ones(time_frames)]
    for power in range(1, 6):
        column_list.append(x_i ** power)
    for j in range(x.shape[1]):
        if i != j:
            x_j = x[:time_frames, j]
            for first_power in range(1, 6):
                for second_power in range(1, 6):
                    column_list.append((x_i ** first_power) * (x_j ** second_power))
    theta = np.column_stack(column_list)
    return theta


def _sindy(x_dot, theta, candidate_lambda):
    xi = np.linalg.lstsq(theta, x_dot, rcond=None)[0]
    for j in range(SINDY_ITERATIONS):
        small_indices = np.flatnonzero(np.absolute(xi) < candidate_lambda)
        big_indices = np.flatnonzero(np.absolute(xi) >= candidate_lambda)
        xi[small_indices] = 0
        xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], x_dot, rcond=None)[0]
    return xi


def run():
    iran_stock_networks = get_iran_stock_networks()
    sorted_iran_stock_networks = sorted(iran_stock_networks, key=lambda network: -network.dynamicity)
    entire_x = sorted_iran_stock_networks[0].x  # TODO test all
    x_rows = []
    x_cv_rows = []
    x_dot_rows = []
    x_dot_cv_rows = []
    for row_index in range(entire_x.shape[0] - 1):
        if row_index % 2 == 1:
            x_rows.append(entire_x[row_index])
            x_dot_rows.append(entire_x[row_index + 1] - entire_x[row_index])
        else:
            x_cv_rows.append(entire_x[row_index])
            x_dot_cv_rows.append(entire_x[row_index + 1] - entire_x[row_index])
    x = np.stack(x_rows)
    x_cv = np.stack(x_cv_rows)
    x_dot = np.stack(x_dot_rows)
    x_dot_cv = np.stack(x_dot_cv_rows)
    x_dot_cv_range = np.max(x_dot_cv) - np.min(x_dot_cv)

    for i in range(entire_x.shape[1]):
        theta = _get_theta(x, i)
        theta_cv = _get_theta(x_cv, i)
        least_cost = sys.maxsize
        best_xi = None
        selected_lambda = 0
        selected_complexity = 0
        selected_mse_cv = 0
        ith_derivative = x_dot[:, i]
        for candidate_lambda in CANDIDATE_LAMBDAS:
            xi = _sindy(ith_derivative, theta, candidate_lambda)
            complexity = np.count_nonzero(xi)  # TODO / np.prod(xi.shape)
            mse_cv = np.square(x_dot_cv[:, i] - (np.matmul(theta_cv, xi.T))).mean()
            if complexity:  # zero would mean no statements
                cost = mse_cv * complexity
                if cost < least_cost:
                    least_cost = cost
                    best_xi = xi
                    selected_lambda = candidate_lambda
                    selected_complexity = complexity
                    selected_mse_cv = mse_cv
        nrmsd = selected_mse_cv ** 0.5 / x_dot_cv_range
        print(nrmsd, selected_complexity)


if __name__ == '__main__':
    run()
