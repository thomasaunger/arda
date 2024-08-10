import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.path import Path
from mpl_toolkits.mplot3d import art3d


def generate_tag_env_rollout_animation(
    trainer=None,
    fps=60,
    power_color="#0000FF",
    angel_color="#FF0000",
    goal_color="#00FF00",
    fig_width=6,
    fig_height=6,
):
    if trainer is not None:
        assert trainer is not None

        episode_states = trainer.fetch_episode_states(
            ["loc_y", "loc_x", "orientations", "goal_point", "rewards", "sampled_actions"]
        )
        assert isinstance(episode_states, dict)
        env = trainer.cuda_envs.env

        space_length = env.space.length
        num_agents = env.num_agents
        episode_length = env.episode_length
    else:
        space_length = 8
        num_agents = 2
        episode_length = 64
        episode_states = {
            "loc_y":        np.random.randint(space_length, size=(episode_length, num_agents)),
            "loc_x":        np.random.randint(space_length, size=(episode_length, num_agents)),
            "orientations": np.random.randint(           4, size=(episode_length, num_agents)),
            "goal_point":   np.random.randint(space_length, size=(episode_length,          2)),
        }

    fig, ax = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  # , constrained_layout=True
    ax.remove()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Bounds
    ax.set_xlim(0.0, space_length)
    ax.set_ylim(0.0, space_length)
    ax.set_zlim(-1.0, 1.0)

    # Space
    for x in range(space_length):
        for y in range(space_length):
            corner_points = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
            color = goal_color if episode_states["goal_point"][0, 1] == x and episode_states["goal_point"][0, 0] == y else (0.1, 0.2, 0.5, 0.15)
            poly = Polygon(corner_points, color=color)
            ax.add_patch(poly)
            art3d.pathpatch_2d_to_3d(poly, z=-0, zdir="z")

    # "Hide" side panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Hide axes
    ax.set_axis_off()

    # Set camera
    ax.elev = 40
    ax.azim = -55
    ax.dist = 10

    # Try to reduce whitespace
    fig.subplots_adjust(left=0, right=1, bottom=-0.2, top=1)

    def state_to_vertices(y, x, orientation):
        match orientation:
            case 0:
                return [(x,     y + 1), (x + 1, y + 1), (x + 0.5, y      ), (x,     y + 1)]
            case 1:
                return [(x,     y    ), (x,     y + 1), (x +   1, y + 0.5), (x,     y    )]
            case 2:
                return [(x + 1, y    ), (x,     y    ), (x + 0.5, y +   1), (x + 1, y    )]
            case 3:
                return [(x + 1, y + 1), (x + 1, y    ), (x,       y + 0.5), (x + 1, y + 1)]

    # Create triangles
    triangles = []
    for idx in range(num_agents):
        # Define vertices of the triangle
        y = episode_states["loc_y"][0, idx]
        x = episode_states["loc_x"][0, idx]
        orientation = episode_states["orientations"][0, idx]
        vertices = state_to_vertices(y, x, orientation)
        if trainer is not None:
            if idx in env.powers:
                color = power_color
            elif idx in env.angels:
                color = angel_color
        else:
            color = power_color if idx == 0 else angel_color
        triangle = Polygon(vertices, color=color)
        ax.add_patch(triangle)
        art3d.pathpatch_2d_to_3d(triangle, z=0, zdir="z")
        triangles.append(triangle)

    label = ax.text(
        0,
        0,
        2.0,
        "gabriel",
    )

    label.set_fontsize(14)
    label.set_fontweight("normal")
    label.set_color("#666666")

    def animate(i):
        for idx, triangle in enumerate(triangles):
            y = episode_states["loc_y"][i, idx]
            x = episode_states["loc_x"][i, idx]
            orientation = episode_states["orientations"][i, idx]

            vertices = state_to_vertices(y, x, orientation)

            # Update the path of the polygon
            # Create a new path with the updated vertices
            path = Path(vertices, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

            # Update the path and z position of the existing PathPatch3D object
            triangle.set_3d_properties(path, zs=0, zdir='z')

        label.set_text(i)

    anim = animation.FuncAnimation(
        fig, animate, np.arange(0, episode_length), interval=1000.0 / fps
    )
    plt.close()

    return anim
