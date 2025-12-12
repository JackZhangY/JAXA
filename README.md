# JAXA

This is a developing project on **JAX**-based **A**gent (**JAXA**) learning framework, where integrates the implementations of various reinforcement learning algorithms based on <a href="https://docs.jax.dev/en/latest/index.html">**JAX**</a> library. 

Tile now, this project mainly includes two types (**Online** / **Offline**) RL algorithms implementations, **Cross-domain (XDomain)** RL algorithms are also on the schedule.

### Planned Implementations of Online RL

<table style="width: 55%;">
  <thead>
    <tr>
      <th width="20%">Online Algs</th>
      <th width="15%">Code File</th>
      <th width="50%">Reference Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><input type="checkbox" checked disabled>DQN</td>
      <td><a href="././agents/dqn_trainer.py">dqn_trainer.py</a></td>
      <td><a href="https://arxiv.org/abs/1312.5602"> Deep Q-Network </a> </td>
    </tr>
    <tr>
      <td><input type="checkbox" checked disabled> AL </td>
      <td><a href="././agents/aldqn_trainer.py">aldqn_trainer.py</a></td>
      <td><a href="https://arxiv.org/abs/1512.04860">Advantage Learning</a></td>
    </tr>
    <tr>
      <td><input type="checkbox" checked disabled>NAF</td>
      <td><a href="././agents/naf_trainer.py">naf_trainer.py</a></td>
      <td><a href="https://arxiv.org/abs/1603.00748">Normalized Advatange Functions</a></td>
    </tr>
    <tr>
      <td><input type="checkbox" checked disabled>clipAL</td>
      <td><a href="././agents/clipaldqn_trainer.py">clipaldqn_trainer.py</a></td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/20900">Clipped Advantage Learning</a></td>
    </tr>
      <td><input type="checkbox" disabled>ORAL</td>
      <td>oraldqn_trainer.py</a></td>
      <td><a href="https://ieeexplore.ieee.org/document/11227018">Occam's Razor-based AL</a></td>
    <tr>
      <td><input type="checkbox" checked disabled>SAC</td>
      <td><a href="././agents/sac_trainer.py">sac_trainer.py</a></td>
      <td><a href="https://arxiv.org/abs/1801.01290">Soft Actor-Critic</a></td>
    </tr>
  </tbody>
</table>

### Planned Implementations of Offline RL

<table style="width: 55%;">
  <thead>
    <tr>
      <th width="25%">Offline Algs</th>
      <th width="25%">Code File</th>
      <th width="50%">Reference Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><input type="checkbox" checked disabled>IQL</td>
      <td><a href="././agents/iql_trainer.py">iql_trainer.py</a></td>
      <td><a href="https://arxiv.org/abs/2110.06169">Implicit Q-Learning</a></td>
    </tr>
    <tr>
      <td><input type="checkbox" disabled>TD3+BC</td>
      <td>td3bc_trainer.py</td>
      <td><a href="https://arxiv.org/pdf/2106.06860">TD3+Behavior Cloing</a></td>
    </tr>
    <tr>
      <td><input type="checkbox" disabled> iTRPO </td>
      <td>itrpo_trainer.py</td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/29637">implicit TRPO</a></td>
    </tr>
    <!-- <tr>
      <td><input type="checkbox" checked disabled>clipAL</td>
      <td><a href="././agents/clipaldqn_trainer.py">clipaldqn_trainer.py</a></td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/20900">Clipped Advantage Learning</a></td>
    </tr>
    <tr>
      <td><input type="checkbox" checked disabled>SAC</td>
      <td><a href="././agents/sac_trainer.py">sac_trainer.py</a></td>
      <td><a href="https://arxiv.org/abs/1801.01290">Soft Actor-Critic</a></td>
    </tr> -->
  </tbody>
</table>
More reinforcement learning algorithms implemented by JAX will be coming soon.
