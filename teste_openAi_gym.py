import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import base64
from IPython import display

def show_video(video_folder):
    """
    Função para exibir vídeos gravados
    """
    html = []
    for video in sorted(video_folder.iterdir()):
        video_base64 = base64.b64encode(video.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(video.name, video_base64.decode('ascii')))
    display.display(display.HTML(data="<br>".join(html)))

# Crie o ambiente
env = DummyVecEnv([lambda: gym.make('Pendulum-v1')])

# Wrapper de vídeo
env = VecVideoRecorder(env, video_folder='./videos/',
                       record_video_trigger=lambda x: x == 0, video_length=500,
                       name_prefix="pendulum-ppo")

# Inicialize o agente PPO
agent = PPO("MlpPolicy", env, verbose=1)

# Treine o agente
agent.learn(total_timesteps=1000)

# Avalie o agente e grave um vídeo
obs = env.reset()
for _ in range(500):
    action, _ = agent.predict(obs)
    obs, _, done, _ = env.step(action)

env.close()
show_video(Path("./videos/"))
