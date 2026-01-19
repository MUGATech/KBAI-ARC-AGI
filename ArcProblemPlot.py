import json
import os
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcColors import arc_colors
from ArcSet import ArcSet


class ArcPlot:

    @staticmethod
    def plot_data(data: ArcData, axis: Axes):
        """
        Plot the Arc Data.
        """
        if data.data() is None:
            return
        arc_grid: np.ndarray = data.data()

        # cmap is the color map to use and vmin/vmax limits the color scale to 10 values (0-9)
        axis.pcolormesh(arc_grid, cmap=arc_colors, vmin=0, vmax=9)
        axis.set_xticks(np.arange(0, arc_grid.shape[1]+1, 1))
        axis.set_yticks(np.arange(0, arc_grid.shape[0]+1, 1))
        axis.grid()
        axis.set_aspect(1)
        axis.invert_yaxis()

    def plot_arc_problem(self, problem: ArcProblem) -> Figure:
        """
        Plot the ArcProblem
        """
        last_idx = len(problem.training_set())
        primary_fig = pyplot.figure(constrained_layout=True, dpi=100)
        primary_fig.suptitle(problem.problem_name())

        grid_spec = GridSpec(nrows=last_idx+1, ncols=2, figure=primary_fig)

        for idx, train in enumerate(problem.training_set()):
            in_sub_fig = primary_fig.add_subfigure(grid_spec[idx, 0])
            in_sub_fig.suptitle(f"Training Data In {idx + 1}")
            in_axis = in_sub_fig.subplots()
            in_data = train.get_input_data()
            self.plot_data(in_data, in_axis)

            out_sub_fig = primary_fig.add_subfigure(grid_spec[idx, 1])
            out_sub_fig.suptitle(f"Training Data Out {idx + 1}")
            out_axis = out_sub_fig.subplots()
            out_data = train.get_output_data()
            self.plot_data(out_data, out_axis)

        in_tst_sub_fig = primary_fig.add_subfigure(grid_spec[last_idx, 0])
        in_tst_sub_fig.suptitle("Test Data Input")
        test = problem.test_set()
        in_tst_axis = in_tst_sub_fig.subplots()
        in_test = test.get_input_data()
        self.plot_data(in_test, in_tst_axis)

        out_tst_sub_fig = primary_fig.add_subfigure(grid_spec[last_idx, 1])
        out_tst_sub_fig.suptitle("Test Data Output")
        out_test = test.get_output_data()
        out_tst_axis = out_tst_sub_fig.subplots()
        self.plot_data(out_test, out_tst_axis)
        return primary_fig


if __name__ == '__main__':
    file_path = filedialog.askopenfilename(initialdir='data', filetypes=[('JSON,', '*.json')],
                                           title='Choose a Problem to Plot')
    file_name = os.path.basename(file_path)
    # opens the file and reads in the data into python objects
    with open(os.path.join(file_path)) as p:
        flat_data: dict[str, dict] = json.load(p)
        trn_data: list[ArcSet] = list()
        for dt in flat_data['train']:
            d_input = ArcData(np.array(dt['input']))
            d_output = ArcData(np.array(dt['output']))
            trn_set: ArcSet = ArcSet(arc_input=d_input, arc_output=d_output)
            trn_data.append(trn_set)

        tst_data: list[ArcSet] = list()
        for tst in flat_data['test']:
            t_input = ArcData(np.array(tst['input']))
            t_output = ArcData(np.array(tst['output']))
            tst_set: ArcSet = ArcSet(arc_input=t_input, arc_output=t_output)
            tst_data.append(tst_set)

        # create the ArcProblem to plot
        arc_problem = ArcProblem(file_name, trn_data, tst_data[0])
        # plot the problem
        figure = ArcPlot().plot_arc_problem(arc_problem)
        figure.set_size_inches(8.5, 14)
        # pyplot.show(block=True)

        # To save the plot as a PDF file, uncomment the next 5 lines

        save_file_name = f'{file_name.strip(".json")}.pdf'
        file_path = filedialog.asksaveasfilename(initialdir='data', initialfile=save_file_name,
                                                 filetypes=[('PDF,', '*.pdf')], title='Choose directory to save file')
        pyplot.savefig(file_path, format='pdf')
        pyplot.close()

