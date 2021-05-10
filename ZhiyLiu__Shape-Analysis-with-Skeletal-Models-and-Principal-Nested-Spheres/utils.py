import numpy as np
import vtk
def vtk_show(renderer, w=1280, h=900):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
#     renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(w, h)
    renderWindow.Render()
     
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
     
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = bytes(memoryview(writer.GetResult()))
    
    from IPython.display import Image
    return Image(data)
def overlay_poly_data(foreground, vague_background):
    """
    Overlay two vtkPolyData
    foreground is shown in blue
    vague_background is shown in transparent grey
    """
    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputData(foreground)
    colors = vtk.vtkNamedColors()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)
    actor.GetProperty().SetColor(colors.GetColor3d("Blue"))
    actor.GetProperty().SetLineWidth(1)

    bk_mapper = vtk.vtkPolyDataMapper()
    bk_mapper.SetInputData(vague_background)
    bk_actor = vtk.vtkActor()
    bk_actor.SetMapper(bk_mapper)
    bk_actor.GetProperty().SetColor(colors.GetColor3d("dim_grey"))
    bk_actor.GetProperty().SetLineWidth(1)
    bk_actor.GetProperty().SetOpacity(0.2)
    ren1 = vtk.vtkRenderer()
    ren1.AddActor(actor)
    ren1.AddActor(bk_actor)

    ren1.SetBackground(colors.GetColor3d("ivory"))

    style = vtk.vtkCamera()
    style.SetPosition(50, 70, 50)
    style.SetFocalPoint(-30, -30, 40)
    ren1.SetActiveCamera(style)
    return ren1
def simulate_data_on_small_circle(num=50):
    """
    Simulate num data points distributed on S^2
    
    """
    theta = np.linspace(0, np.pi * 1.5, num)
    signal = 0.25*np.concatenate((np.cos(theta)[None, :], np.sin(theta)[None, :]))
    noise = np.random.randn(2,num) * 0.05
    data = signal + noise
    
    rotMat = rotate_to_north_pole([1,2,0], np.pi/3)
    data = np.matmul(rotMat.T, exp_north_pole(data))
    return data
    
def is_on_unit_sphere(data):
    """
    Return true if data (of dimension d x n) are distributed on a unit sphere
    """
    d, n = data.shape
    for i in range(n):
        if abs(np.linalg.norm(data[:, i]) - 1) > 1e-8:
            return False
    return True

def rotate_to_north_pole(v, angle=None):
    """
    Rotate a unit vector v to the north pole of a unit sphere

    Return the rotation matrix
    See Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    Input: 1D array (i.e., a point on a unit sphere)
    """
    d = len(v) # dimension of the feature space

    ## normalize vector v
    v = v / np.linalg.norm(v)
    ## north pole coordinates d-dimension [0, ..., 0, 1]
    north_pole = [0] * (d - 1)
    north_pole.append(1)
    north_pole = np.asarray(north_pole)

    inner_prod = np.inner(north_pole, v)
    if not angle:
        angle = np.arccos(inner_prod)
    if np.abs(inner_prod - 1) < 1e-15:
        return np.eye(d)
    elif np.abs(inner_prod + 1) < 1e-15:
        return -np.eye(d)
    c = v - north_pole * inner_prod
    c = c / np.linalg.norm(c)
    A = np.outer(north_pole, c) - np.outer(c, north_pole)

    rot = np.eye(d) + np.sin(angle)*A + (np.cos(angle) - 1)*(np.outer(north_pole, north_pole)\
                                                             + np.outer(c, c))
    return rot
def log_north_pole(x):
    """
    LOGNP Riemannian log map at North pole of S^k
        LogNP(x) returns k x n matrix where each column is a point on tangent
        space at north pole and the input x is (k+1) x n matrix where each column
        is a point on a sphere.
    Input: d x n matrix w.r.t. the extrinsic coords system
    Output: (d-1) x n matrix w.r.t. the coords system (tangent space) origined at the NP
    """
    d, n = x.shape
    scale = np.arccos(x[-1, :]) / np.sqrt(1-x[-1, :]**2)
    scale[np.isnan(scale)] = 1
    log_px = scale * x[:-1, :]
    return log_px
def exp_north_pole(x):
    """
    EXPNP Riemannian exponential map at North pole of S^k
    returns (k+1) x n matrix where each column is a point on a
    sphere and the input v is k x n matrix where each column
    is a point on tangent  space at north pole.

    Input: d x n matrix
    """
    d, n = x.shape
    nv = np.sqrt(np.sum(x ** 2, axis=0))
    tmp = np.sin(nv) / (nv + 1e-15)
    exp_px = np.vstack((tmp * x, np.cos(nv)))
    exp_px[:, nv < 1e-16] = np.repeat(np.vstack((np.zeros((d, 1)), 1)), np.sum(nv<1e-16), axis=1)
    return exp_px
def viz_directions_distribution(pts, pts2, mu1=None, mu2=None, pts_joint=None, title=''):
    """
    Draw directional data distributed on a unit sphere
    Input pts: n x 3
    Input pts2: nx 3
    Input pts_joint: the bending directions of pts n x 3
    Input mu1/mu2: 1 x 3
    """
    assert pts.shape[0] == pts2.shape[0], "Two blocks have to be of same number of samples"
    ### draw unit sphere
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    elev = 10.0
    rot = 80.0 / 180 * np.pi
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, cmap=cm.Greys, linewidth=0, alpha=0.3)
    #calculate vectors for "vertical" circle
    a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    b = np.array([0, 1, 0])
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
    ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = 'dashed')
    horiz_front = np.linspace(0, np.pi, 100)
    ax.plot(np.sin(horiz_front),np.cos(horiz_front),0,color='k')
    ax.view_init(elev = elev, azim = 0)

    for i in range(pts.shape[0]):
        px, py, pz = pts[i, :]

        px2, py2, pz2 = pts2[i, :]


        ax.scatter(px, py, pz, c='r')
        ax.scatter(px2, py2, pz2, c='b', marker='v')
        if pts_joint is not None:
            px_j, py_j, pz_j = pts_joint[i, :]
            ax.scatter(px_j, py_j, pz_j, c='y', marker='*')

    if mu1 is not None:
        ax.scatter(mu1[0], mu1[1], mu1[2], c='r', marker='x')
    if mu2 is not None:
        ax.scatter(mu2[0], mu2[1], mu2[2], c='b', marker='x')
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    
def drawCircleS2(center,theta):
    """
    draw circle on unit sphere S2 by center of a small circle
    converted code of Sungkyu Jung MATLAB drawCircleS2.m
    """
    if(theta==np.pi):
        t=np.c_[np.cos(np.linspace(start=0, stop=2*np.pi, num=50)),
                np.sin(np.linspace(start=0, stop=2*np.pi, num=50)),np.cos(theta)*np.repeat(0, 50)]
        smallCircle=np.matmul(t,rotate_to_north_pole(center))
    else:
        t=np.c_[np.sin(theta)*np.cos(np.linspace(start=0, stop=2*np.pi, num=50)),
                np.sin(theta)*np.sin(np.linspace(start=0, stop=2*np.pi, num=50)),
                np.cos(theta)*np.repeat(1, 50)]
        smallCircle=np.matmul(t,rotate_to_north_pole(center))
    return smallCircle