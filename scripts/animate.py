from matplotlib import animation

from realm.utils import generate_tag_env_rollout_animation


anim = generate_tag_env_rollout_animation(fps=6)
writergif = animation.PillowWriter(fps=6)
anim.save("realm.gif", writer=writergif)
