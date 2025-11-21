#!/usr/bin/python

"""
This file contains a method to compute mesh quality parameters. You should always use this after making a mesh.
"""

################################################################################
##   Project:   FEniCS CFD Simulation Setup / Mesh Quality Assessment
##   Date:      2017/08/21 04:05:44
##   Version:   1.0
##   Author:    Mehdi Najafi, (mnuoft at gmail). All rights reserved.
##
##
##   This script is distributed WITHOUT ANY WARRANTY; without even the implied 
##   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
##
################################################################################

import os, sys
import math
import vtk
import numpy
from collections import OrderedDict

# ----------------------------------------------------------------------
def getNominalA(h):
    return 0.25 * (3**0.5) * h*h
# ----------------------------------------------------------------------
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        if filename:
            self.log = open(filename, 'w')
        else:
            self.log = None

    def __del__(self):
        if self.log:
            self.log.close()
    
    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass



# =====================================================================
class MeshDisplay3D:
    def __init__(self, title, _dir):
        self.Data = None
        self._Surface = None
        self._CaseName = None
        self._path = _dir
        self._SeedIds = None
        self._TargetSeedIds = vtk.vtkIdList()
        self.PickedSeedIds = vtk.vtkIdList()
        self.PickedSeeds = vtk.vtkPolyData()
        self.vtkRenderer = None
        self.KeyBindings = {}
        self.Script = None
        self.selectedpointindex = 0
        self.SeedActors = []
        self.TextActor = None
        self.TextInputMode = 0
        self.ExitAfterTextInputMode = True
        self.ExitTextInputCallback = None
        self.surfaceActor = None
        self.surfaceOpacity = 1.0
        self.selectedMapper = None
        self.selectedActor = None
        self.selectedCellsCount = 0

    def HighlightCells(self, lstCells):
        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        for id in lstCells:
            ids.InsertNextValue(id)

        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(ids)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extractSelection = vtk.vtkExtractSelection()
        if vtk.VTK_MAJOR_VERSION <= 5:
            extractSelection.SetInput(0, self.Data)
            extractSelection.SetInput(1, selection)
        else:
            extractSelection.SetInputData(0, self.Data)
            extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        # In selection
        selected = vtk.vtkUnstructuredGrid()
        selected.ShallowCopy(extractSelection.GetOutput())

        self.selectedMapper = vtk.vtkDataSetMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.selectedMapper.SetInputConnection(selected.GetProducerPort())
        else:
            self.selectedMapper.SetInputData(selected)

        self.selectedActor = vtk.vtkActor()
        self.selectedActor.SetMapper(self.selectedMapper)
        self.selectedActor.GetProperty().EdgeVisibilityOn()
        self.selectedActor.GetProperty().SetEdgeColor(1.0, 0.6, 0.0)
        self.selectedActor.GetProperty().SetLineWidth(2)
        #self.selectedActor.GetProperty().SetVertexColor(0.8, 0.48, 0.0)

        self.glyph = vtk.vtkGlyph3D()
        glyphSource = vtk.vtkArrowSource()
        glyphSource.InvertOn()
        self.glyph.SetSourceConnection(glyphSource.GetOutputPort())
        self.glyph.SetInputData(selected)
        #self.glyph.SetColorModeToColorByScalar()
        #self.glyph.SetScaleModeToDataScalingOff() 
        #self.glyph.SetVectorModeToUseVector()
        self.glyph.SetScaleFactor(self._Surface.GetLength()*0.012)
        self.glyph.Update()

        self.gylphmapper = vtk.vtkPolyDataMapper()
        self.gylphmapper.SetInputData(self.glyph.GetOutput())
        
        self.gylphactor = vtk.vtkActor()
        self.gylphactor.SetMapper(self.gylphmapper)
        self.gylphactor.VisibilityOff()
        self.gylphactor.GetProperty().SetOpacity(0.15)
        self.gylphactor.GetProperty().SetColor(0.8, 0.48, 0.0)

        self.selectedCellsCount = len(lstCells)


    def SetSurface(self, surface, casename):
        self._Surface = surface
        self._CaseName = casename
        self.Data = self._Surface


    def GetSurface(self):
        return self._Surface

    def GetTargetSeedIds(self):
        return self._TargetSeedIds

    def GetText(self):
        if self.selectedCellsCount>0:
            if self.selectedCellsCount == 1:
                text = 'Please rotate the model to observe a cell with difficulties.'
            else:
                text = 'Please rotate the model to observe %d cells with difficulties.'%(self.selectedCellsCount)
        else:
            text = 'The model looks fine. You may press \'q\' to quit.'
        return text

    def ToggleHelpCallback(self, obj):
        self.Text.SetVisibility(1-self.Text.GetVisibility())
        self.TextActor.SetVisibility(1-self.TextActor.GetVisibility())
        self.TextInputActor.SetVisibility(1-self.TextInputActor.GetVisibility())
        self.RenderWindow.Render()

    def ToggleSelectedCellsCallback(self, obj):
        self.selectedActor.SetVisibility(1-self.selectedActor.GetVisibility())
        self.RenderWindow.Render()

    def ToggleGylphSelectedCellsCallback(self, obj):
        self.gylphactor.SetVisibility(1-self.gylphactor.GetVisibility())
        self.RenderWindow.Render()

    def ToggleSurfaceCallback(self, obj):
        self.surfaceActor.SetVisibility(1-self.surfaceActor.GetVisibility())
        self.RenderWindow.Render()

    def IncSurfaceOpacityCallback(self, obj):
        self.surfaceOpacity += 0.1
        if self.surfaceOpacity > 1:
            self.surfaceOpacity = 1
        self.surfaceActor.GetProperty().SetOpacity(self.surfaceOpacity)
        self.RenderWindow.Render()
    def DecSurfaceOpacityCallback(self, obj):
        self.surfaceOpacity -= 0.1
        if self.surfaceOpacity < 0:
            self.surfaceOpacity = 0
        self.surfaceActor.GetProperty().SetOpacity(self.surfaceOpacity)
        self.RenderWindow.Render()

    def SolidEdgeModeCallback(self, obj):
        self.surfaceActor.GetProperty().SetRepresentationToSurface()
        self.surfaceActor.GetProperty().EdgeVisibilityOn()#(1-self.surfaceActor.GetProperty().GetEdgeVisibility())
        self.RenderWindow.Render()
    def SolidModeCallback(self, obj):
        self.surfaceActor.GetProperty().SetRepresentationToSurface()
        self.surfaceActor.GetProperty().EdgeVisibilityOff()
        self.RenderWindow.Render()
    def WireframeModeCallback(self, obj):
        self.surfaceActor.GetProperty().SetRepresentationToWireframe()
        self.surfaceActor.GetProperty().EdgeVisibilityOff()
        self.RenderWindow.Render()

    def ScreenshotCallback(self, obj):
        self.ShowAllText(False)
        filePrefix = self._CaseName+'-meshquality-screenshot'
        fileNumber = 0
        fileName = "%s-%d.png" % (filePrefix,fileNumber)
        existingFiles = os.listdir(os.path.dirname(self._CaseName))
        while fileName in existingFiles:
            fileNumber += 1
            fileName = "%s-%d.png" % (filePrefix,fileNumber)
        #fileName = self._path + fileName
        print('Saving screenshot to ' + fileName)
        windowToImage = vtk.vtkWindowToImageFilter()
        windowToImage.SetInput(self.RenderWindow)
        windowToImage.SetMagnification(4)
        windowToImage.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(windowToImage.GetOutputPort())
        writer.SetFileName(fileName)
        writer.Write()
        self.ShowAllText()

    def ShowAllText(self, on=True):
        self.Text.SetVisibility(on)
        self.TextInputActor.SetVisibility(on)
        self.TextActor.SetVisibility(on)
        self.RenderWindow.Render()

    def QuitRendererCallback(self, obj):
        #print('User quit request - 3D View terminated.')
        self.CleanUp()
        self.RenderWindowInteractor.ExitCallback()

    def ResetCameraCallback(self, obj):
        self.vtkRenderer.ResetCamera()
        self.RenderWindow.Render()

    def UpdateTextInput(self):
        if self.TextInputQuery:
            if self.CurrentTextInput or self.CurrentTextInput == '':
                self.TextInputActor.SetInput(self.TextInputQuery+self.CurrentTextInput+'_')
            else:
                self.TextInputActor.SetInput(self.TextInputQuery)
            self.vtkRenderer.AddActor(self.TextInputActor)
        else:
            self.vtkRenderer.RemoveActor(self.TextInputActor)
        self.RenderWindow.Render()

    def KeyPressCallback(self, obj, event):
        key = self.RenderWindowInteractor.GetKeySym()
        if key =='Escape':
            if self.TextInputMode:
                self.TextInputMode = 0
            else:
                self.TextInputMode = 1
        if self.TextInputMode:
            if key in ['Return','Enter']:
                self.ExitTextInputMode()
                return
            if key.startswith('KP_'):
                key = key[3:]
            if key == 'space':
                key = ' '
            elif key in ['minus','Subtract']:
                key = '-'
            elif key in ['period','Decimal']:
                key = '.'
            elif len(key) > 1 and key not in ['Backspace','BackSpace']:
                key = None
            if key in ['Backspace','BackSpace']:
                textInput = self.CurrentTextInput
                if len(textInput) > 0:
                    self.CurrentTextInput = textInput[:-1]
            elif key:
                self.CurrentTextInput += key
            self.UpdateTextInput()
            return

        if key in self.KeyBindings and self.KeyBindings[key]['callback'] != None:
            self.KeyBindings[key]['callback'](obj)
        else:
            if key == 'plus':
                key = '+'
            if key == 'minus':
                key = '-'
            if key == 'equal':
                key = '='
            if key in self.KeyBindings and self.KeyBindings[key]['callback'] != None:
                self.KeyBindings[key]['callback'](obj)

    def AddKeyBinding(self, key, text, callback=None, group='1'):
        self.KeyBindings[key] = {'text': text, 'callback': callback, 'group': group}

    def RemoveKeyBinding(self, key):
        if key in self.KeyBindings:   
            del self.KeyBindings[key]
            
    def InitializeSeeds(self):
        self.PickedSeedIds.Initialize()
        self.PickedSeeds.Initialize()
        seedPoints = vtk.vtkPoints()
        self.PickedSeeds.SetPoints(seedPoints)

    def CleanUp(self):
        self.vtkRenderer.RemoveActor(self.TextActor)
        self.vtkRenderer.RemoveActor(self.TextInputActor)

    def ShowHelpText(self):
        groups = list(set([self.KeyBindings[el]['group'] for el in self.KeyBindings]))
        #groups.sort(reverse=True)
        textActorInputsList = []
        for group in groups:
            sortedKeys = [key for key in self.KeyBindings.keys() if self.KeyBindings[key]['group'] == group]
            sortedKeys.sort()
            textActorInputs = ['[%s]: %s' % (key, self.KeyBindings[key]['text']) for key in sortedKeys]
            textActorInputsList.append('\n'.join(textActorInputs))
        self.TextActor.SetInput('\n\n'.join(textActorInputsList))
        #self.RenderWindow.Render()

    def ShowKnownTypeHelp(self):
        self.ShowHelpText()
        self.updateTitle() 


    def updateTitle(self):
        prefix = self._CaseName
        self.RenderWindow.SetWindowName(prefix + " - Mesh Inspection Tool :: BioMedical Simulations Lab")

    def Execute(self):

        if (self._Surface == None):
            print('Error: No input Surface set.')
            return

        self._TargetSeedIds.Initialize()

        if not self.vtkRenderer:
            self.vtkRenderer = vtk.vtkRenderer()
            self.vtkRenderer.SetBackground(0.15, 0.49, 0.37) #0.37, 0.49)
            if self.selectedCellsCount > 0:
                self.vtkRenderer.SetBackground(0.15, 0.25, 0.27) #0.37, 0.49)
            
            self.RenderWindow = vtk.vtkRenderWindow()
            self.updateTitle()
            self.RenderWindow.AddRenderer(self.vtkRenderer)
            self.RenderWindow.SetSize(800,600)
            self.RenderWindow.SetPosition(10,10)
            self.RenderWindow.SetPointSmoothing(1)
            self.RenderWindow.SetLineSmoothing(1)
            self.RenderWindow.SetPolygonSmoothing(0)
            self.RenderWindowInteractor = vtk.vtkRenderWindowInteractor()
            self.RenderWindow.SetInteractor(self.RenderWindowInteractor)
            self.RenderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.RenderWindowInteractor.GetInteractorStyle().KeyPressActivationOff()
            self.RenderWindowInteractor.GetInteractorStyle().AddObserver("KeyPressEvent",self.KeyPressCallback)

            self.AddKeyBinding('x','Take Screenshot',self.ScreenshotCallback,'0')
            self.AddKeyBinding('r','Reset Camera',self.ResetCameraCallback,'0')
            self.AddKeyBinding('w','Wireframe Mode',self.WireframeModeCallback,'0')
            self.AddKeyBinding('s','Solid Mode', self.SolidModeCallback,'0')
            self.AddKeyBinding('a','Solid+Edge Mode', self.SolidEdgeModeCallback,'0')
            self.AddKeyBinding('q','Quit/Ignore',self.QuitRendererCallback,'0')
            self.AddKeyBinding('c','Toggle Odd Cells',self.ToggleSelectedCellsCallback,'0')
            self.AddKeyBinding('g','Toggle Geometry',self.ToggleSurfaceCallback, '0')
            self.AddKeyBinding('m','Toggle Markers',self.ToggleGylphSelectedCellsCallback, '0')
            self.AddKeyBinding('h','Toggle Help',self.ToggleHelpCallback, '0')
            self.AddKeyBinding('+','Increase Opacity',self.IncSurfaceOpacityCallback, '0')
            self.AddKeyBinding('-','Decrease Opacity',self.DecSurfaceOpacityCallback, '0')

            self.TextActor = vtk.vtkTextActor()
            self.TextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            self.TextActor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            self.TextActor.SetPosition([0.005, 0.1])
            self.TextActor.GetTextProperty().SetFontSize(15)
            self.TextActor.GetTextProperty().SetFontFamilyToCourier()
            self.TextActor.GetTextProperty().SetColor(0.6,0.6,0.6)

            self.ShowHelpText()
            self.vtkRenderer.AddActor(self.TextActor)

            self.TextInputActor = vtk.vtkTextActor()
            self.TextInputActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            self.TextInputActor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            self.TextInputActor.SetPosition([0.25, 0.1])

            self.vtkRenderer.AddActor(self.TextInputActor)

            self.ShowKnownTypeHelp()


        # create a text actor
        self.Text = vtk.vtkTextActor()
        self.Text.SetInput(self.GetText())
        txtprop=self.Text.GetTextProperty()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(15)
        #txtprop.SetColor(0.8,0.9,0.3)
        txtprop.SetColor(0.9,0.9,0.9)
        self.Text.SetDisplayPosition(20,10)
        self.vtkRenderer.AddActor(self.Text)

        #surfaceMapper = vtk.vtkPolyDataMapper()

        #surfaceMapper.SetInputData(self._Surface)
        #surfaceMapper.ScalarVisibilityOff()

        surfaceMapper = vtk.vtkDataSetMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            surfaceMapper.SetInputConnection(self._Surface.GetProducerPort())
        else:
            surfaceMapper.SetInputData(self._Surface)


        self.surfaceActor = vtk.vtkActor()
        self.surfaceActor.SetMapper(surfaceMapper)
        self.surfaceActor.GetProperty().SetOpacity(1.0)
        self.surfaceActor.GetProperty().SetEdgeColor(0.0, 0.2, 0.6)

        self.vtkRenderer.AddActor(self.surfaceActor)

        if self.selectedActor:
            self.vtkRenderer.AddActor(self.selectedActor)
            self.vtkRenderer.AddActor(self.gylphactor)

        self.InitializeSeeds()
        
        self.RenderWindowInteractor.Initialize()
        self.RenderWindow.Render()

        self.RenderWindowInteractor.Start()

        #self.PickedSeedIds.GetNumberOfIds()

        self._TargetSeedIds.DeepCopy(self.PickedSeedIds)

        self.RenderWindowInteractor = None
        self.RenderWindow = None
        self.vtkRenderer = None 
# =====================================================================
    

# ----------------------------------------------------------------------
def check(value):
    if abs(value) < 1E-6 or abs(value) > 1E2:
        return False
    return True

# ----------------------------------------------------------------------
def MeshQualityMeasures(title, polyDataVolMesh, outputfilename = '', 
                        only_check_surface = False, display_cells = False):
    if polyDataVolMesh == None:
        return -1

    out = Logger(outputfilename)

    quality = vtk.vtkMeshQuality()
    vtk_version = vtk.vtkVersion().GetVTKMajorVersion()
    if vtk_version < 6:
        quality.SetInput(polyDataVolMesh)
    else:
        quality.SetInputData(polyDataVolMesh)
    quality.SaveCellQualityOn()
#    quality.SetTriangleQualityMeasureToRadiusRatio()
#    quality.Update()

    # Here we define the various mesh types and labels for output.
    meshTypes = [
                 ['Triangle', 'Triangle',
                  [
                   ['QualityMeasureToArea', 'Area'],
                   ['QualityMeasureToEdgeRatio', 'Edge Ratio'],
                   ['QualityMeasureToAspectRatio', 'Aspect Ratio'],
                   ['QualityMeasureToRadiusRatio', 'Radius Ratio'],
                   ['QualityMeasureToAspectFrobenius', 'Frobenius Norm'],
                   ['QualityMeasureToMinAngle', 'Min Angle']
                   ]
                  ],

                 ['Tet', 'Tetrahedron',
                  [
                   ['QualityMeasureToVolume', 'Volume'],
                   ['QualityMeasureToEdgeRatio', 'Edge Ratio'],
                   ['QualityMeasureToAspectRatio', 'Aspect Ratio'],
                   ['QualityMeasureToRadiusRatio', 'Radius Ratio'],
                   ['QualityMeasureToAspectFrobenius', 'Frobenius Norm'],
                   ['QualityMeasureToMinAngle', 'Min Dihedral Angle'],
                   ['QualityMeasureToCollapseRatio', 'Collapse Ratio'],
                   ['QualityMeasureToScaledJacobian', 'Scaled Jacobian'],
                   ['QualityMeasureToJacobian', 'Positive Jacobian'],
                   ['QualityMeasureToJacobian', 'Negative Jacobian']
                   ]
                 ]]

    # remove volume check if surface triangles are to be checked
    if only_check_surface:
        del meshTypes[-1]

        edge = []
        for i in range(polyDataVolMesh.GetNumberOfCells()):
            npts = polyDataVolMesh.GetCell(i).GetPoints().GetNumberOfPoints()
            for k in range(npts):
                point0 = [0.0, 0.0, 0.0]
                point1 = [0.0, 0.0, 0.0]
                polyDataVolMesh.GetCell(i).GetPoints().GetPoint(k, point0)
                polyDataVolMesh.GetCell(i).GetPoints().GetPoint((k+1)%npts, point1)
                d = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1))
                edge.append(d)
        min = numpy.amin(edge)
        max = numpy.amax(edge)
        average = numpy.mean(edge)
        median = numpy.median(edge)
        stddev = numpy.std(edge)
        q75, q25 = numpy.percentile(edge, [75 ,25])
        irqrange = '%7.5g,%7.5g'%(q25,q75)
        outliers = [ [[],[]], [[],[]], [[],[]], [[],[]], [[],[]] ]
        for qe in edge:
            if abs(average - qe) > 4 * stddev:
                if qe < average:
                    outliers[4][0].append(qe)
                else:
                    outliers[4][1].append(qe)
            elif abs(average - qe) > 3 * stddev:
                if qe < average:
                    outliers[3][0].append(qe)
                else:
                    outliers[3][1].append(qe)
            elif abs(average - qe) > 2 * stddev:
                if qe < average:
                    outliers[2][0].append(qe)
                else:
                    outliers[2][1].append(qe)
            elif abs(average - qe) > 1 * stddev:
                if qe < average:
                    outliers[1][0].append(qe)
                else:
                    outliers[1][1].append(qe)
        len_outliers = []
        for i in range(len(outliers)):
            len_outliers.append(len(outliers[i][0]) + len(outliers[i][1]))
        len_out = '%8d|%8d|%8d'%(len_outliers[2],len_outliers[3],len_outliers[4])

        edge_line = ' %6s|%19s|%12.5g|%12.5g|%12.5g|%12.5g|%12.5g|%23s|%23s\n'%('CHECK','EdgeLength',min,max,median,average,stddev,irqrange,len_out)
        edge_line += ' %6s|%19s|%12.5g|%12.5g|%12.5g|%12.5g|%12.5g|%23s|%23s\n'%('CHECK','NominalArea',getNominalA(min),getNominalA(max),getNominalA(median),getNominalA(average),stddev,irqrange,len_out)
    else:
        edge_line = ''


    disp = OrderedDict()

    good_mesh = 1
    if polyDataVolMesh.GetNumberOfCells() > 0:
        res = ''
        out.write ('*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***\n')
        out.write ('*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***\n')
        out.write ('--- Mesh Quality Check : \n')
        out.write ('Number of cells: %i, Number of points: %i\n' % (polyDataVolMesh.GetNumberOfCells(), polyDataVolMesh.GetNumberOfPoints()))
        num_cells = polyDataVolMesh.GetNumberOfCells()
        if (num_cells > 6000000):
            out.write (f' \n WARNING: the number of cells is suspiciously high - {num_cells}. \n ')
        elif num_cells < 1000: 
            out.write (f' \n WARNING: the number of cells is suspiciously low - {num_cells}. \n ')
        for meshType in meshTypes:
            out.write ('\n')
            eval('quality.Set' + meshType[0] + meshType[2][0][0])
            quality.Update()
            statQuality = quality.GetOutput().GetFieldData().GetArray('Mesh ' + meshType[1] + ' Quality')
            n = int(statQuality.GetComponent(0, 4))

            out.write (' -------------------------------------------- \n')
            out.write ('  Mesh Element/Cell: %10d%12ss |\n'%(n,meshType[1]))
            out.write (' ----------------------------------------------------------------------------------------------------------------------------------------------\n')
            out.write (' status|      measure      |    min     |    max     |   median   |   average  |   stddev   |  interquartile range  |    #outliers: 2, 3, 4    \n')
            out.write (' ------+-------------------+------------+------------+------------+------------+------------+-----------------------+--------+--------+--------\n')
            if edge_line:
                out.write(edge_line)
            for measure in meshType[2]:
                eval('quality.Set' + meshType[0] + measure[0] + '()')
                quality.Update()
                statQuality = quality.GetOutput().GetFieldData().GetArray('Mesh ' + meshType[1] + ' Quality')
                n = int(statQuality.GetComponent(0, 4))
                qArray = quality.GetOutput().GetCellData().GetArray('Quality')

                q = []
                min = 1E30
                max = -min
                for i in range(n):
                    v = qArray.GetValue(i)
                    if measure[1]=='Volume' or measure[1]=='Area':
                        v = abs(v)
                    if measure[1]=='Positive Jacobian' and v<0:
                        pass
                    elif measure[1]=='Negative Jacobian' and v>0:
                        pass
                    else:
                        q.append(v)

                    if check(v) == False:
                        # add i to the list of cells to be displayed in different color
                        disp[i] = True

                if len(q) == 0:
                    continue
                min = numpy.amin(q)
                max = numpy.amax(q)
                average = numpy.mean(q)
                median = numpy.median(q)
                stddev = numpy.std(q)

                q75, q25 = numpy.percentile(q, [75 ,25])
                irqrange = '%7.5g,%7.5g'%(q25,q75)
                outliers = [ [[],[]], [[],[]], [[],[]], [[],[]], [[],[]] ]
                for qe in q:
                    if abs(average - qe) > 4 * stddev:
                        if qe < average:
                            outliers[4][0].append(qe)
                        else:
                            outliers[4][1].append(qe)
                    elif abs(average - qe) > 3 * stddev:
                        if qe < average:
                            outliers[3][0].append(qe)
                        else:
                            outliers[3][1].append(qe)
                    elif abs(average - qe) > 2 * stddev:
                        if qe < average:
                            outliers[2][0].append(qe)
                        else:
                            outliers[2][1].append(qe)
                    elif abs(average - qe) > 1 * stddev:
                        if qe < average:
                            outliers[1][0].append(qe)
                        else:
                            outliers[1][1].append(qe)
                len_outliers = []
                for i in range(len(outliers)):
                    len_outliers.append(len(outliers[i][0]) + len(outliers[i][1]))
                len_out = '%8d|%8d|%8d'%(len_outliers[2],len_outliers[3],len_outliers[4])

                status = '  OK  '

                if len_outliers[3] > n/10:
                    status = 'fair?'

                if (measure[1]=='Area') and (average < 1E-6 or min < 1E-6):
                    status = ' !!!! '

                if (measure[1]=='Volume') and (average < 1E-6 or min < 1E-7):
                    status = ' !!!! '

                if measure[1].rfind('Jacobian') > 0 and (abs(min) < 2E-6 or abs(max) < 2E-6):
                    status = ' !!!! '

                if (max > 500):
                    status = ' !!!! '

                if status.find(' OK ') < 0:
                    good_mesh = -1

                # VTK limitation trap
                if meshType[0] == 'Triangle' and only_check_surface == False:
                    min = statQuality.GetComponent(0, 0)
                    max = statQuality.GetComponent(0, 2)
                    average = statQuality.GetComponent(0, 1)
                    median = 'NoDataAccess'
                    stddev = math.sqrt(statQuality.GetComponent(0, 3))
                    line = ' %6s|%19s|%12.5g|%12.5g|%12s|%12.5g|%12.5g|%23s|%23s\n'%(status,measure[1],min,max,median,average,stddev,median,median)
                else:
                    line = ' %6s|%19s|%12.5g|%12.5g|%12.5g|%12.5g|%12.5g|%23s|%23s\n'%(status,measure[1],min,max,median,average,stddev,irqrange,len_out)
                out.write (line)
            out.write ('\n')

        out.write ('*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***\n')

        if outputfilename:
            file = open(outputfilename,'w')
            res.replace('\n','\r\n')
            file.write(res)
            file.close()
    else:
        raise RuntimeError('The mesh has no cells.')

    if display_cells:
        # display a 3D enviroment with those cells in different colors
        dispWnd = MeshDisplay3D(title, os.path.dirname(outputfilename))
        dispWnd.SetSurface(polyDataVolMesh, title)
        dispWnd.HighlightCells(disp)
        dispWnd.Execute()

    return good_mesh


if __name__ == "__main__":
        surface_check_only = False

        fileName = sys.argv[1]
        ## Load the given file, and pass the vtkPolyData object
        fileType = fileName[-3:]
        if fileType == '':
                raise RuntimeError('The file does not have an extension')
        if fileType == 'stl':
                reader = vtk.vtkSTLReader()
                reader.MergingOn()
                surface_check_only = True
        elif fileType == 'vtk':
                reader = vtk.vtkPolyDataReader()
        elif fileType == 'vtp':
                reader = vtk.vtkXMLPolyDataReader()
        elif fileType == 'vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
        else:
                raise RuntimeError('Unknown file type %s' % fileType)
        reader.SetFileName(fileName)
        reader.Update()
        polyData = reader.GetOutput()

        outputfilename = os.path.splitext(sys.argv[1])[0]+'_mesh_quality.txt'

        if MeshQualityMeasures(fileName, polyData, outputfilename, surface_check_only, False) < 0:
            print ('NOT a good mesh! Be careful.')
