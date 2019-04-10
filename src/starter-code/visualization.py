import plotly as py
from plotly.offline import plot, iplot
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


class Visualization(object):
    def __init__(self):
        pass

    def simpleplt(self, input_arr):
        # plotting the image
        plt.imshow(input_arr, cmap="gray")

    def visualize(
        self,
        object_data_frame,
        view,
        x_range=[-30, 30],
        y_range=[-120, 120],
        use_distance=False,
    ):
        x = tuple(object_data_frame["X"].tolist())
        y = tuple(object_data_frame["Y"].tolist())
        z = tuple(object_data_frame["Z"].tolist())
        if use_distance:
            radius = tuple(object_data_frame["radius"].tolist())
        else:
            radius = "red"
        if view == 1:
            x_points = x
            y_points = z
        elif view == 2:
            x_points = x
            y_points = y
        elif view == 3:
            x_points = z
            y_points = y
        else:
            print("Not a valid view")
            x_points = x
            y_points = z
        trace = go.Scatter(
            x=x_points,
            y=y_points,
            mode="markers",
            marker=dict(color=radius, size=2, opacity=1),
        )
        layout = go.Layout(
            scene=dict(xaxis=dict(range=x_range), yaxis=dict(range=y_range))
        )

        data = [trace]
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)

    def visualize_50_html(self, num_of_scenes, view, objects, object_name):
        """This function creates the visualization of the object points in particular view for given num_of_scenes."""
        # Creating the list of scenes
        scenes = range(num_of_scenes)

        # make figure
        figure = {"data": [], "layout": {}, "frames": []}

        # fill in most of layout
        figure["layout"]["xaxis"] = {"range": [-120, 120], "title": "x-axis"}
        figure["layout"]["yaxis"] = {"title": "y-axis"}
        figure["layout"]["hovermode"] = "closest"
        figure["layout"]["sliders"] = {
            "args": ["transition", {"duration": 400, "easing": "cubic-in-out"}],
            "initialValue": "0",
            "plotlycommand": "animate",
            "values": scenes,
            "visible": True,
        }
        figure["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 300,
                                    "easing": "quadratic-in-out",
                                },
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Scene:",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }

        # make data
        scene = 0

        object_by_scene = objects[scene]
        x = tuple(object_by_scene["X"].tolist())
        y = tuple(object_by_scene["Y"].tolist())
        z = tuple(object_by_scene["Z"].tolist())
        radius = tuple(object_by_scene["radius"].tolist())
        if view == 1:
            x_points = x
            y_points = z

        elif view == 2:
            x_points = x
            y_points = y

        elif view == 3:
            x_points = z
            y_points = y

        else:
            print("Not a valid view")
            x_points = x
            y_points = z

        data_dict = {
            "x": x_points,
            "y": y_points,
            "mode": "markers",
            "marker": {"color": radius, "size": 2, "opacity": 1},
        }
        figure["data"].append(data_dict)

        # make frames
        for scene in scenes:
            frame = {"data": [], "name": str(scene)}

            object_by_scene = objects[scene]
            x = tuple(object_by_scene["X"].tolist())
            y = tuple(object_by_scene["Y"].tolist())
            z = tuple(object_by_scene["Z"].tolist())
            radius = tuple(object_by_scene["radius"].tolist())
            if view == 1:
                x_points = x
                y_points = z
            elif view == 2:
                x_points = x
                y_points = y
            elif view == 3:
                x_points = z
                y_points = y
            else:
                print("Not a valid view")
                x_points = x
                y_points = z

            data_dict = {
                "x": x_points,
                "y": y_points,
                "mode": "markers",
                "marker": {"color": radius, "size": 2, "opacity": 1},
            }
            frame["data"].append(data_dict)

            figure["frames"].append(frame)

            slider_step = {
                "args": [
                    [scene],
                    {
                        "frame": {"duration": 300, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": scene,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        figure["layout"]["sliders"] = [sliders_dict]

        plot(
            figure,
            filename="../visuals/{}_{}_{}.html".format(
                object_name, num_of_scenes, view
            ),
            auto_open=False,
            show_link=False,
        )

    def plot_grid(self, object_points, grid_size, view, x_range, y_range):
        """Function to plot the given object points in the given view with grid"""
        # plots only for two views
        y = tuple(object_points["Y"].tolist())

        if view == 2:
            x = tuple(object_points["X"].tolist())
        elif view == 3:
            x = tuple(object_points["Z"].tolist())

        # Plotting the figure
        fig = plt.figure(figsize=(20, 20), dpi=80, facecolor="w", edgecolor="k")
        ax = fig.gca()
        ax.set_xticks(np.arange(x_range[0], x_range[1], grid_size))
        ax.set_yticks(np.arange(y_range[0], y_range[1], grid_size))
        plt.scatter(x, y, s=4)
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        plt.grid()
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix, cls_pred, cls_true, num_classes):

        # get the confusion matrix using sklearn
        cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

        # Printing the confusion matrix as text
        print(cm)

        # plot the confusion matrix as image
        plt.matshow(cm)

        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.show()
