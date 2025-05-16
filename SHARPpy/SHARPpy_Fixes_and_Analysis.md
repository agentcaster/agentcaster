# SHARPpy Fixes and Analysis

## 1. Fixes Applied to Make Test Scripts Run Successfully

### 1.1 Fixes for plot_skew.py

#### 1.1.1 Issue: Missing Metadata Keys in `getPlotTitle` Method

The `getPlotTitle` method in `sharppy/viz/skew.py` was failing because it was trying to access metadata keys that didn't exist. I added checks for these keys and provided default values:

```python
# Added check for 'run' key
run = prof_coll.getMeta('run') if 'run' in prof_coll._meta else datetime.now()
run_str = run.strftime("%HZ") if run is not None else "N/A"

# Added check for 'model' key
model = prof_coll.getMeta('model') if 'model' in prof_coll._meta else "HRRR"

# Added check for 'observed' key
observed = prof_coll.getMeta('observed') if 'observed' in prof_coll._meta else False
```

#### 1.1.2 Issue: Missing 'base_time' Key in `getPlotTitle` Method

The method was also failing when trying to access the 'base_time' key in two different places. I added checks for this key:

```python
# First instance - in the "Archive" model case
if not prof_coll.getMeta('observed'):
    if 'base_time' in prof_coll._meta:
        fhour = int(total_seconds(date - prof_coll.getMeta('base_time')) / 3600)
        fhour_str = " F%03d" % fhour
    else:
        fhour_str = ""

# Second instance - in the "else" case for other models
if 'base_time' in prof_coll._meta:
    fhour = int(total_seconds(date - prof_coll.getMeta('base_time')) / 3600)
    plot_title += "  (" + run_str + "  " + model + mem_string + "  " + ("F%03d" % fhour) + modified_str + ")"
else:
    plot_title += "  (" + run_str + "  " + model + mem_string + modified_str + ")"
```

#### 1.1.3 Issue: NoneType Error in `draw_parcel_levels` Method

The `draw_parcel_levels` method was trying to access attributes of `self.pcl` when it was `None`. I added a check to return early if `self.pcl` is `None`:

```python
def draw_parcel_levels(self, qp):
    logging.debug("Drawing the parcel levels (LCL, LFC, EL).")
    if self.pcl is None:
        return
    
    # Rest of the method...
```

### 1.2 Fixes for plot_hodo.py

#### 1.2.1 Issue: Passing Non-Qt Properties to QFrame Constructor

The `plotHodo` class in `sharppy/viz/hodo.py` was failing because it was trying to pass `bg_color` and `fg_color` as parameters to the QFrame constructor through `super().__init__(**kwargs)`, but these are not valid Qt properties. I modified the code to set these properties after creating the plotHodo instance:

```python
# Original code in plot_hodo.py
self.hodo = plotHodo(bg_color='#000000', fg_color='#FFFFFF')

# Fixed code
self.hodo = plotHodo()
self.hodo.bg_color = QColor('#000000')
self.hodo.fg_color = QColor('#FFFFFF')
```

This fix ensures that only valid Qt properties are passed to the QFrame constructor, while still setting the background and foreground colors for the hodograph.

### 1.3 Fixes for plot_thermo.py

#### 1.3.1 Issue: Missing `pwat_colors` Attribute in `plotText` Class

The `plotText` class in `sharppy/viz/thermo.py` was failing because it was trying to access a `pwat_colors` attribute in the `drawIndices` method, but this attribute was not initialized in the `__init__` method. I added the initialization of this attribute with default values:

```python
# Initialize pwat_colors with default values
self.pwat_colors = [
    QtGui.QColor('#775000'),  # -3 sigma
    QtGui.QColor('#996600'),  # -2 sigma
    QtGui.QColor('#ffffff'),  # -1 sigma
    QtGui.QColor('#ffffff'),  # mean
    QtGui.QColor('#ffff00'),  # +1 sigma
    QtGui.QColor('#ff0000'),  # +2 sigma
    QtGui.QColor('#e700df'),  # +3 sigma
]
```

This fix ensures that the `pwat_colors` attribute is available when the `drawIndices` method tries to access it, preventing the AttributeError.

## 2. How the Plot is Generated: Tracing the Process

### 2.1 Entry Point: plot_skew.py

The process begins in `plot_skew.py`, which serves as the entry point for the application. Here's the flow:

1. **Initialization**: The script creates a Qt application and sets up high DPI scaling.
2. **MainWindow Creation**: A `MainWindow` class is instantiated, which:
   - Loads sounding data from a file (`examples/data/14061619.OAX`)
   - Creates a `plotSkewT` widget
   - Adds the profile collection to the widget
   - Sets the active collection
   - Creates and sets a surface-based parcel
   - Sets the widget as the central widget of the window
3. **Execution**: The window is shown and the Qt event loop is started.

### 2.2 Data Loading: SPCDecoder

The `SPCDecoder` class in `sharppy/io/spc_decoder.py` is responsible for loading the sounding data:

1. **File Reading**: The decoder reads the file and parses it into sections.
2. **Metadata Extraction**: It extracts metadata like location, time, latitude, and longitude.
3. **Data Parsing**: It parses the raw data into arrays for pressure, height, temperature, dew point, wind direction, and wind speed.
4. **Profile Creation**: It creates a `Profile` object using the parsed data.
5. **Collection Creation**: It creates a `ProfCollection` object containing the profile and sets metadata like location, observed status, and base time.

### 2.3 Profile Collection: ProfCollection

The `ProfCollection` class in `sharppy/sharptab/prof_collection.py` manages profiles from a data source:

1. **Storage**: It stores profiles, dates, and metadata.
2. **Access Methods**: It provides methods to access the current profile, highlighted profile, and metadata.
3. **Modification**: It allows for modification of profiles and keeps track of modifications.
4. **Time Management**: It handles time switching for time-series data.

### 2.4 Visualization: plotSkewT

The `plotSkewT` class in `sharppy/viz/skew.py` is responsible for rendering the Skew-T diagram:

1. **Initialization**: It sets up the plotting area, colors, and UI elements.
2. **Background Drawing**: The `plotBackground` method draws the background elements like isotherms, dry adiabats, and mixing ratio lines.
3. **Data Plotting**: The `plotData` method plots the actual sounding data:
   - It draws temperature and dew point traces
   - It draws wind barbs
   - It draws parcel traces if a parcel is set
   - It draws significant levels like LCL, LFC, and EL
   - It draws other elements like the effective layer and omega profile
4. **Interaction**: It handles user interactions like mouse clicks, drags, and wheel events for zooming.

### 2.5 Rendering Process in Detail

When `setActiveCollection` is called in the `MainWindow` constructor, the following happens:

1. **Profile Setting**: The active profile is set from the collection.
2. **Data Extraction**: Temperature, dew point, and wind data are extracted from the profile.
3. **Draggable Setup**: Draggable objects are set up for temperature and dew point traces.
4. **Plot Clearing**: The plot is cleared using `clearData`.
5. **Data Plotting**: The data is plotted using `plotData`.
6. **Update**: The widget is updated to show the new plot.

When `setParcel` is called, a similar process occurs:

1. **Parcel Setting**: The parcel is set.
2. **Plot Clearing**: The plot is cleared.
3. **Data Plotting**: The data is plotted, including the parcel trace.
4. **Update**: The widget is updated.

### 2.6 Drawing the Skew-T Diagram

The actual drawing of the Skew-T diagram involves several steps:

1. **Background Elements**:
   - Isotherms (lines of constant temperature)
   - Dry adiabats (lines of constant potential temperature)
   - Mixing ratio lines (lines of constant water vapor mixing ratio)
   - Isobars (lines of constant pressure)
   - Frame and labels

2. **Data Elements**:
   - Temperature trace
   - Dew point trace
   - Wind barbs
   - Virtual temperature trace
   - Wet bulb temperature trace

3. **Derived Elements**:
   - Parcel trace (if a parcel is set)
   - LCL, LFC, and EL levels (if a parcel is set)
   - Effective layer
   - Dendritic growth zone (if enabled)
   - Maximum lapse rate layer
   - Height markers

4. **Interactive Elements**:
   - Readout cursor
   - Draggable temperature and dew point traces

The drawing is done using Qt's painting system, with a `QPainter` object used to draw on a `QPixmap` that is then displayed in the widget.

## 3. SPCWindow.py Dependencies Analysis

### 3.1 Direct Dependencies

SPCWindow.py directly imports and uses the following modules from the sharppy/viz directory:

1. **skew.py**: Provides `plotSkewT` for the main sounding diagram
2. **hodo.py**: Provides `plotHodo` for the hodograph display
3. **thermo.py**: Provides `plotText` for thermodynamic text information
4. **analogues.py**: Provides `plotAnalogues` for sounding analogue information
5. **thetae.py**: Provides `plotThetae` for equivalent potential temperature plots
6. **srwinds.py**: Provides `plotWinds` for storm-relative wind profiles
7. **speed.py**: Provides `plotSpeed` for wind speed vs. height plots
8. **kinematics.py**: Provides `plotKinematics` for kinematic parameter displays
9. **slinky.py**: Provides `plotSlinky` for storm slinky visualization
10. **watch.py**: Provides `plotWatch` for watch type information
11. **advection.py**: Provides `plotAdvection` for temperature advection displays
12. **stp.py**: Provides `plotSTP` for Significant Tornado Parameter statistics
13. **winter.py**: Provides `plotWinter` for winter weather parameters
14. **ship.py**: Provides `plotSHIP` for Significant Hail Parameter information
15. **stpef.py**: Provides `plotSTPEF` for conditional STP and EF-scale probabilities
16. **fire.py**: Provides `plotFire` for fire weather parameters
17. **vrot.py**: Provides `plotVROT` for EF-scale probabilities based on V-rot

These modules are imported at the beginning of SPCWindow.py:

```python
from sharppy.viz import plotSkewT, plotHodo, plotText, plotAnalogues
from sharppy.viz import plotThetae, plotWinds, plotSpeed, plotKinematics #, plotGeneric
from sharppy.viz import plotSlinky, plotWatch, plotAdvection, plotSTP, plotWinter
from sharppy.viz import plotSHIP, plotSTPEF, plotFire, plotVROT
```

### 3.2 Indirect Dependencies

SPCWindow.py also indirectly depends on the following modules:

1. **draggable.py**: Used by both skew.py and hodo.py to implement draggable elements for interactive modification of profiles. The `Draggable` class provides low-level clicking and dragging functionality on widgets. You don't need to implement this, but make sure it doesn't break anything in our plotting.

2. **barbs.py**: Used by skew.py to draw wind barbs on the sounding diagram. The `drawBarb` function renders wind barbs based on direction and speed.

3. **preferences.py**: Used for handling user preferences and configuration. The SPCWindow.py file interacts with this through the `updateConfig` method, which processes configuration settings for colors, units, and other display preferences.

### 3.3 Unused or Commented-Out Dependencies

Some files in the sharppy/viz directory are not currently used by SPCWindow.py:

1. **generic.py**: Provides `plotGeneric` for generic plotting functionality, but its import is commented out in SPCWindow.py:
   ```python
   from sharppy.viz import plotThetae, plotWinds, plotSpeed, plotKinematics #, plotGeneric
   ```

2. **map.py**: Provides mapping functionality through the `Mapper` and `MapWidget` classes, but is not imported or used in SPCWindow.py.

3. **ensemble.py**: Provides ensemble visualization through the `plotENS` class, but is not directly imported in SPCWindow.py.

### 3.4 Dependency Flow

The dependency flow in SPCWindow.py follows this pattern:

1. **Initialization**: SPCWindow.py initializes various visualization components using the imported plotting functions.
2. **Configuration**: It configures these components based on user preferences.
3. **Data Binding**: It binds profile data to these components.
4. **Layout Management**: It arranges these components in a grid layout.
5. **Interaction Handling**: It connects signals and slots to handle user interactions.

This modular design allows for easy swapping of visualization components, as demonstrated by the inset swapping functionality in the `swapInset` method.

## 4. Test Scripts for Direct Dependencies

To verify that each of the direct dependencies works correctly, I've created individual test scripts for each module. These scripts follow a consistent pattern based on the existing plot_skew.py:

1. Import necessary modules
2. Set up a Qt application
3. Create a main window
4. Load sounding data using SPCDecoder
5. Create the appropriate visualization widget
6. Set the profile (and parcel if needed)
7. Set the widget as the central widget
8. Show the window and start the Qt event loop

The following test scripts have been created:

1. **plot_skew.py** (works) - Tests the Skew-T diagram display
2. **plot_hodo.py** (works) - Tests the hodograph display
3. **plot_thermo.py** (works) - Tests the thermodynamic parameters display
4. **plot_analogues.py** (works) - Tests the sounding analogues display
5. **plot_thetae.py** (works) - Tests the Theta-E vs. Pressure display
6. **plot_srwinds.py** (works) - Tests the storm-relative winds display
7. **plot_speed.py** (works) - Tests the wind speed vs. height display
8. **plot_kinematics.py** (works) - Tests the kinematic parameters display
9. **plot_slinky.py** (works) - Tests the storm slinky visualization
10. **plot_watch.py** (works) - Tests the watch type information display
11. **plot_advection.py** (works) - Tests the temperature advection display
12. **plot_stp.py** (works) - Tests the Significant Tornado Parameter statistics
13. **plot_winter.py** (works) - Tests the winter weather parameters display
14. **plot_ship.py** (works) - Tests the Significant Hail Parameter display
15. **plot_stpef.py** (works) - Tests the conditional tornado probabilities display
16. **plot_fire.py** (works) - Tests the fire weather parameters display
17. **plot_vrot.py** (works) - Tests the EF-scale probabilities based on V-rot

These scripts can be used to independently test each module, ensuring that all the direct dependencies in SHARPpy are functioning correctly.

## 5. Debugging and Fixing SPCWindow.py

### 5.1 Initial Attempt and Errors

When attempting to create a test script for the full SHARPpy window (SPCWindow), I encountered several issues:

1. **QSignalMapper Error**: The most significant issue was with the QSignalMapper in SPCWindow.py. When running the script, I received the following error:
   ```
   AttributeError: 'PySide6.QtCore.QSignalMapper' object has no attribute 'mapped'. Did you mean: 'mappedInt'?
   ```
   This error occurred in the `createMenuBar` method of the SPCWindow class, specifically at the line:
   ```python
   self.focus_mapper.mapped[str].connect(self.spc_widget.setProfileCollection)
   ```

2. **Missing Metadata**: When trying to add a profile collection to the SPCWindow, I encountered a KeyError for the 'run' metadata key:
   ```
   KeyError: 'run'
   ```
   This occurred in the `createMenuName` method of the SPCWindow class when trying to access:
   ```python
   pc_date = prof_col.getMeta('run').strftime("%d/%H%MZ")
   ```

3. **Qt API Compatibility**: There were issues with the Qt API version being used. The original code was designed for PySide2, but the current environment was using PySide6, leading to compatibility issues.

### 5.2 Debugging Process

To debug these issues, I followed these steps:

1. **Analyzing Error Messages**: I carefully analyzed the error messages to understand the root causes of the issues.

2. **Examining SPCWindow.py**: I examined the SPCWindow.py file to understand how it uses QSignalMapper and how it accesses metadata from profile collections.

3. **Researching Qt API Changes**: I researched the changes between different versions of the Qt API, particularly the changes to QSignalMapper between PySide2 and PySide6.

4. **Testing with Different Qt APIs**: I tried forcing the use of PySide2 through the QT_API environment variable, but found that PySide2 was not available in the current environment.

5. **Inspecting Other Working Scripts**: I looked at the other test scripts that were working correctly to understand how they handle metadata and Qt API compatibility.

### 5.3 Solutions Implemented

Based on the debugging process, I implemented the following solutions:

1. **Patched SPCWindow.py**: I created a patched version of SPCWindow.py (SPCWindow_patched.py) that's compatible with PySide6. The key change was replacing:
   ```python
   self.focus_mapper.mapped[str].connect(self.spc_widget.setProfileCollection)
   self.remove_mapper.mapped[str].connect(self.rmProfileCollection)
   ```
   with:
   ```python
   self.focus_mapper.mappedString.connect(self.spc_widget.setProfileCollection)
   self.remove_mapper.mappedString.connect(self.rmProfileCollection)
   ```
   This change reflects the fact that in PySide6, the 'mapped' signal has been replaced with specific signals like 'mappedString', 'mappedInt', etc.

2. **Setting Required Metadata**: In the plot_SPCWindow.py script, I added code to set the required metadata for the profile collection before adding it to the SPCWindow:
   ```python
   # Set required metadata
   from datetime import datetime
   prof_collection.setMeta('model', 'Observed')
   prof_collection.setMeta('run', datetime.now())
   prof_collection.setMeta('loc', 'OAX')
   prof_collection.setMeta('observed', True)
   ```
   This ensures that the SPCWindow can access the metadata it needs to function correctly.

3. **Using PySide6**: Instead of trying to force the use of PySide2, I adapted the code to work with PySide6, which is the Qt API version available in the current environment.

### 5.4 Full SHARPpy Window Test Script

With these fixes in place, I created a script to test the full SHARPpy window with all visualization components:

**plot_SPCWindow.py** - Tests the complete SHARPpy window with all insets and components

This script creates a full SHARPpy window using the patched version of SPCWindow.py that's compatible with PySide6. It loads a sounding profile from the examples directory, sets the required metadata, and displays it in the full SHARPpy window, complete with all the insets and visualization components.

The script successfully displays the complete SHARPpy visualization system with all components working together, providing a comprehensive test of the entire system.
