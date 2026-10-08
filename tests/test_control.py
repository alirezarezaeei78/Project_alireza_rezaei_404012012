import math
import unittest
import numpy as np
import torch
from pendulum_control.environment import BallStabilizeEnv,Config
from pendulum_control.controllers import PIDController
from pendulum_control.dqn import double_dqn_target,ReplayBuffer,Transition


class ControlTests(unittest.TestCase):
    def test_compiled_actuator_range_and_rate_limit(self):
        env=BallStabilizeEnv()
        np.testing.assert_allclose([env.lo,env.hi],np.deg2rad([-170,170]))
        env.step_target(999)
        self.assertAlmostEqual(env.command_deg,1.2)
        self.assertLessEqual(env.data.ctrl[env.actuator],env.hi)

    def test_observation_is_read_only_and_reset_repeats(self):
        env=BallStabilizeEnv();a=env.observation();b=env.observation()
        np.testing.assert_array_equal(a,b)
        self.assertLess(np.linalg.norm(env.bias[:2]),1e-8)
        first=env.step(1)[0];env.reset();second=env.step(1)[0]
        np.testing.assert_array_equal(first,second)

    def test_disturbance_precedes_success_and_time_limit_truncates(self):
        env=BallStabilizeEnv(config=Config(max_steps=1))
        _,_,terminated,truncated,_=env.step(0)
        self.assertFalse(terminated);self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):env.step(0)

    def test_pid_derivative_uses_control_interval(self):
        pid=PIDController(.005,kp=0,ki=0,kd=.001,deadband=0)
        pid.target(0)
        self.assertAlmostEqual(pid.target(1),-math.degrees(.2))

    def test_pid_anti_windup_at_both_limits(self):
        for error in [-10,10]:
            pid=PIDController(.005,kp=1,ki=1,kd=0,deadband=0)
            for _ in range(20):pid.target(error)
            self.assertEqual(pid.integral,0.)

    def test_double_dqn_selects_online_evaluates_target(self):
        rewards=torch.tensor([1.,1.]);terminal=torch.tensor([1.,0.])
        online=torch.tensor([[1.,2.],[3.,1.]]);target=torch.tensor([[9.,4.],[5.,8.]])
        result=double_dqn_target(rewards,terminal,online,target,gamma=.9)
        torch.testing.assert_close(result,torch.tensor([1.,5.5]))

    def test_replay_sampling_after_capacity_wrap(self):
        rb=ReplayBuffer(capacity=2)
        for i in range(3):rb.push(Transition(np.array([i],dtype=np.float32),0,0.,np.array([i+1],dtype=np.float32),False))
        states,*_=rb.sample(2)
        self.assertEqual(set(states.flatten()),{1,2})


if __name__=='__main__':unittest.main()
