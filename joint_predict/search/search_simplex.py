import sys
import scipy
import numpy as np
from joint_predict.search.search_base import SearchBase

from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing


class SearchSimplex(SearchBase):

    def __init__(
            self,
            search_method=None,
            eval_method="default",
            optimize_method="default",
            budget=100,
            prediction_limit=50,
            random_state=None
    ):
        super().__init__(search_method, eval_method, budget, prediction_limit, random_state)

        self.optimize_func = None

        if optimize_method is None or optimize_method == "default" or optimize_method == "Nelder-Mead":
            self.optimize_func = self.optimize_nm
        elif optimize_method == "Nelder-Mead-4":
            self.optimize_func = self.optimize_nm4
        elif optimize_method == "Double-Annealing":
            self.optimize_func = self.optimize_da

    def search(self, jps):
        super().search(jps)
        k = self.prediction_limit
        # Make sure we don't go over the number of predictions we have
        k = min(k, len(self.pred_indices) - 1)
        best_result = sys.float_info.max
        best_params = None

        # Iterate over all prediction indices
        for prediction_index in range(k):
            # We optimize with flip on and off
            optimize_result = self.optimize_nm(jps, prediction_index, False)
            optimize_flip_result = self.optimize_nm(jps, prediction_index, True)
            # Then take the best result
            final_optimize_result = optimize_result
            flip = False
            if optimize_flip_result.fun < optimize_result.fun:
                final_optimize_result = optimize_flip_result
                flip = True

            if final_optimize_result.fun < best_result:
                offset = final_optimize_result.x[0]
                if self.cache[prediction_index]["skip_rotation"]:
                    rotation = 0
                else:
                    rotation = final_optimize_result.x[1]
                best_result = final_optimize_result.fun
                x = [offset, rotation]
                # Compute the transform again
                # from the parameters to keep track of it
                transform = self.get_transform_from_x(
                    x,
                    jps,
                    prediction_index,
                    flip,
                    self.cache[prediction_index]["align_mat"],
                    self.cache[prediction_index]["origin2"],
                    self.cache[prediction_index]["direction2"]
                )
                _, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method)
                best_params = {
                    "prediction_index": prediction_index,
                    "offset": final_optimize_result.x[0],
                    "rotation": rotation,
                    "flip": flip,
                    "transform": transform,
                    "evaluation": best_result,
                    "overlap": overlap,
                    "contact": contact,
                }
        return best_params

    def search_single(self, jps, prediction_index=0):
        """
        Search
        """
        super().search(jps)
        k = self.prediction_limit
        # Make sure we don't go over the number of predictions we have
        k = min(k, len(self.pred_indices) - 1)
        assert prediction_index <= k

        self.times = 0

        optimize_result = self.optimize_func(jps, prediction_index, False)

        self.times = 0

        optimize_flip_result = self.optimize_func(jps, prediction_index, True)

        final_optimize_result = optimize_result
        flip = False
        if optimize_flip_result.fun < optimize_result.fun:
            final_optimize_result = optimize_flip_result
            flip = True

        transform = self.get_transform_from_x(final_optimize_result.x, jps, prediction_index, flip,
                                              self.cache[prediction_index]["align_mat"],
                                              self.cache[prediction_index]["origin2"],
                                              self.cache[prediction_index]["direction2"]
                                              )

        _, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method, name="Best all")

        return transform, final_optimize_result, overlap, contact

    def search_best(self, jps):
        super().search(jps)
        k = self.prediction_limit
        # Make sure we don't go over the number of predictions we have
        k = min(k, len(self.pred_indices) - 1)
        best_result = sys.float_info.max
        best_transform = None

        # Iterate over all prediction indices
        for prediction_index in range(k):
            self.times = 0
            # We optimize with flip on and off
            optimize_result = self.optimize_nm(jps, prediction_index, False)
            optimize_flip_result = self.optimize_nm(jps, prediction_index, True)
            # Then take the best result
            final_optimize_result = optimize_result
            flip = False
            if optimize_flip_result.fun < optimize_result.fun:
                final_optimize_result = optimize_flip_result
                flip = True

            if final_optimize_result.fun < best_result:
                offset = final_optimize_result.x[0]
                if self.cache[prediction_index]["skip_rotation"]:
                    rotation = 0
                else:
                    rotation = final_optimize_result.x[1]
                best_result = final_optimize_result.fun
                x = [offset, rotation]
                # Compute the transform again
                # from the parameters to keep track of it
                best_transform = self.get_transform_from_x(
                    x,
                    jps,
                    prediction_index,
                    flip,
                    self.cache[prediction_index]["align_mat"],
                    self.cache[prediction_index]["origin2"],
                    self.cache[prediction_index]["direction2"]
                )

        _, overlap, contact = self.env.evaluate(jps, best_transform, eval_method=self.eval_method, name=f"Best all")
        return best_transform

    def optimize_nm(self, jps, prediction_index, flip):
        """Run the optimization"""
        args = (
            jps,
            prediction_index,
            flip,
            self.cache[prediction_index]["align_mat"],
            self.cache[prediction_index]["origin2"],
            self.cache[prediction_index]["direction2"]
        )

        bounds = scipy.optimize.Bounds([-1.2, -190], [1.2, 190])
        initial_guess = np.array([0, 0])
        initial_simplex = np.array([
                    [initial_guess[0] - 0.2, initial_guess[1] - 15],  # 初始猜测点
                    [initial_guess[0] + 0.2, initial_guess[1]],  # 沿偏移方向增加步长
                    [initial_guess[0], initial_guess[1] + 15]  # 沿旋转方向增加步长
                ])
        # print("nm:")
        return scipy.optimize.minimize(
            self.cost_function,
            initial_guess,
            args,
            method="Nelder-Mead",
            # Note: Need scipy 1.7 otherwise:
            # Method Nelder-Mead cannot handle constraints nor bounds
            bounds=bounds,
            options={
                'xatol': 1e-2,
                'fatol': 1e-3,
                "disp": False,
                "maxiter": self.budget,
                "initial_simplex": initial_simplex
            }
        )

    def optimize_nm4(self, jps, prediction_index, flip):
        """Run the optimization"""
        args = (
            jps,
            prediction_index,
            flip,
            self.cache[prediction_index]["align_mat"],
            self.cache[prediction_index]["origin2"],
            self.cache[prediction_index]["direction2"]
        )

        if self.cache[prediction_index]["skip_rotation"]:
            print("skip_rotation!")
            bounds = scipy.optimize.Bounds([-1.2], [1.2])
            initial_guess = np.array([0])

            # initial_simplex = np.array([[initial_guess[0]], [initial_guess[0] + 0.0002]])
            initial_simplex = np.array([
                [initial_guess[0] - 0.2],
                [initial_guess[0] + 0.2],
            ])

            return scipy.optimize.minimize(
                self.cost_function,
                initial_guess,
                args,
                method="Nelder-Mead",
                # Note: Need scipy 1.7 otherwise:
                # Method Nelder-Mead cannot handle constraints nor bounds
                bounds=bounds,
                options={
                    'xatol': 1e-2,
                    'fatol': 1e-3,
                    "disp": False,
                    "maxiter": self.budget,
                    "initial_simplex": initial_simplex
                }
            )
        else:
            bounds = [
                scipy.optimize.Bounds([-1.2, -50], [1.2, 50]),
                scipy.optimize.Bounds([-1.2, 40], [1.2, 140]),
                scipy.optimize.Bounds([-1.2, 130], [1.2, 230]),
                scipy.optimize.Bounds([-1.2, 220], [1.2, 320])
            ]
            initial_guesses = [
                np.array([0, 0]),
                np.array([0, 90]),
                np.array([0, 180]),
                np.array([0, 270])
            ]
            best_result = None
            for initial_guess, bound in zip(initial_guesses, bounds):

                initial_simplex = np.array([
                    [initial_guess[0] - 0.2, initial_guess[1] - 15],  # 初始猜测点
                    [initial_guess[0] + 0.2, initial_guess[1]],  # 沿偏移方向增加步长
                    [initial_guess[0], initial_guess[1] + 15]  # 沿旋转方向增加步长
                ])

                result = scipy.optimize.minimize(
                    self.cost_function,
                    initial_guess,
                    args,
                    method="Nelder-Mead",
                    bounds=bound,
                    options={
                        'xatol': 1e-2,
                        'fatol': 1e-3,
                        "disp": False,
                        "maxiter": self.budget,
                        "initial_simplex": initial_simplex
                    }
                )

                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            return best_result

    def optimize_da(self, jps, prediction_index, flip):
        """Run the optimization"""
        args = (
            jps,
            prediction_index,
            flip,
            self.cache[prediction_index]["align_mat"],
            self.cache[prediction_index]["origin2"],
            self.cache[prediction_index]["direction2"]
        )
        if self.cache[prediction_index]["skip_rotation"]:
            bounds = [(-1.2, 1.2)]
        else:
            bounds = [[-1.2, 1.2], [-190, 190]]

        # print(f"da")
        result = dual_annealing(
            self.cost_function,
            bounds,
            args=args,
            maxiter=self.budget,
        )

        return result

    def cost_function(self, x, jps, prediction_index, flip, align_mat, origin2, direction2):
        """Cost function for use with scipy.optimize"""
        self.times += 1
        transform = self.get_transform_from_x(x, jps, prediction_index, flip, align_mat, origin2, direction2)
        # Evaluate how optimal the parameters are by passing the transform
        # returns a value in the 0-1 range where lower is better
        # print(transform)
        # print(f"{self.times}: ")
        if flip:
            result, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method,
                                                         len=self.cache[prediction_index]["offset_limit"],
                                                         name=f"flip_{self.times}")
        else:
            result, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method,
                                                         len=self.cache[prediction_index]["offset_limit"],
                                                         name=f"noflip_{self.times}")

        # print(f"No. {self.times}: {x}, {result}")
        return result

    def get_transform_from_x(self, x, jps, prediction_index, flip, align_mat, origin2, direction2):
        """Get the transform when given x containing the optimization parameters"""
        offset_limit = self.cache[prediction_index]["offset_limit"]
        # x increments by very tiny amounts
        # scale it to something sensible
        offset = x[0] * offset_limit

        if len(x) == 2:
            rotation = x[1]
        else:
            rotation = 0

        # print(f"x: {x}")
        # print(f"offset: {offset}", end=", ")
        # print(f"rotation: {rotation}")

        return self.env.get_transform_from_parameters(
            jps,
            prediction_index=prediction_index,
            offset=offset,
            rotation_in_degrees=rotation,
            flip=flip,
            align_mat=align_mat,
            origin2=origin2,
            direction2=direction2
        )
