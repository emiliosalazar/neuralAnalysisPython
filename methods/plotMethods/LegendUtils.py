# will contain classes that lets you manipulate the legend
from matplotlib.legend_handler import HandlerLine2D, HandlerPatch
from matplotlib.lines import Line2D

class TransitionHandler(HandlerLine2D):
    def __init__(self, marker_pad=0.3, numpoints=None, stNamesInOrder=None, transitionColors=None, **kw):
        HandlerLine2D.__init__(self, marker_pad=marker_pad, numpoints=numpoints, **kw)
        assert len(stNamesInOrder) - 1 == transitionColors.shape[0], "Must have one fewer transitions than states!"
        self._stNmOrd = stNamesInOrder
        self._transCol = transitionColors

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
#        [legline, legline_marker] = HandlerLine2D.create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        yCenterLoc = (ydescent+height)/2

        legline_marker = Line2D(xdata_marker, [yCenterLoc])
        self.update_prop(legline_marker, orig_handle, legend)
        legline_marker.set_linestyle('None')
        if legend.markerscale != 1:
            newsz = legline_marker.get_markersize() * legend.markerscale
            legline_marker.set_markersize(newsz)

        handLab = orig_handle.get_label()
        currLabLoc = (handLab == self._stNmOrd).nonzero()[0][0]
        legline = []
        if (currLabLoc != self._stNmOrd.shape[0]-1):
            leglineNew = Line2D([xdata_marker[0], xdata_marker[0]], [yCenterLoc, -0.75*height], color = self._transCol[currLabLoc], solid_capstyle='butt', linewidth=1)
            leglineNew.set_marker("")
            leglineNew._legmarker = legline_marker

            leglineNew.set_transform(trans)
            legline += [leglineNew]

        if (currLabLoc != 0):
#            leglineNew = plt.annotate("", xy = (height, yCenterLoc), xytext = (xdata_marker[0], xdata_marker[0]), arrowprops=dict(arrowstyle="->", ec=self._transCol[currLabLoc-1]))
            leglineNew = Line2D([xdata_marker[0], xdata_marker[0]], [1.75*height, yCenterLoc], color = self._transCol[currLabLoc-1], solid_capstyle='butt', linewidth=1)
            leglineNew.set_marker("")
            leglineNew._legmarker = legline_marker

            leglineNew.set_transform(trans)
            legline += [leglineNew]

#        self.update_prop(leglineNew, orig_handle, legend)
#        leglineNew.set_drawstyle('default')
        legline_marker.set_transform(trans)
        return [legline_marker] + legline

# class ColorByTimeHandler(HandlerLine2D):
#     def __init__(self, marker_pad=0.3, numpoints=None, colormapRange = None, **kw):
#         HandlerLine2D.__init__(self, marker_pad=marker_pad, numpoints=numpoints, **kw)
#         self._colormapRange = colormapRange

#     def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
# #        [legline, legline_marker] = HandlerLine2D.create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
#         xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
#                                              width, height, fontsize)

#         yCenterLoc = (ydescent+height)/2

#         legline_marker = Line2D(xdata_marker, [yCenterLoc])
#         self.update_prop(legline_marker, orig_handle, legend)
#         legline_marker.set_linestyle('None')
#         if legend.markerscale != 1:
#             newsz = legline_marker.get_markersize() * legend.markerscale
#             legline_marker.set_markersize(newsz)

        

#         handLab = orig_handle.get_label()
#         currLabLoc = (handLab == self._stNmOrd).nonzero()[0][0]
#         legline = []
#         if (currLabLoc != self._stNmOrd.shape[0]-1):
#             leglineNew = Line2D([xdata_marker[0], xdata_marker[0]], [yCenterLoc, -0.75*height], color = self._transCol[currLabLoc], solid_capstyle='butt', linewidth=1)
#             leglineNew.set_marker("")
#             leglineNew._legmarker = legline_marker

#             leglineNew.set_transform(trans)
#             legline += [leglineNew]

#         if (currLabLoc != 0):
# #            leglineNew = plt.annotate("", xy = (height, yCenterLoc), xytext = (xdata_marker[0], xdata_marker[0]), arrowprops=dict(arrowstyle="->", ec=self._transCol[currLabLoc-1]))
#             leglineNew = Line2D([xdata_marker[0], xdata_marker[0]], [1.75*height, yCenterLoc], color = self._transCol[currLabLoc-1], solid_capstyle='butt', linewidth=1)
#             leglineNew.set_marker("")
#             leglineNew._legmarker = legline_marker

#             leglineNew.set_transform(trans)
#             legline += [leglineNew]

# #        self.update_prop(leglineNew, orig_handle, legend)
# #        leglineNew.set_drawstyle('default')
#         legline_marker.set_transform(trans)
        # return [legline_marker] + legline
