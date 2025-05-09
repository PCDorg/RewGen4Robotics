class Walker2dEnv(MujocoEnv, utils.EzPickle):
   
        def _get_obs(self):
            position = self.data.qpos.flatten()
            velocity = np.clip(self.data.qvel.flatten(), -10, 10)

            if self._exclude_current_positions_from_observation:
                position = position[1:]

            observation = np.concatenate((position, velocity)).ravel()
            return observation