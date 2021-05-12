import vtk
import numpy as np

def fit_srep(obj_mesh, standard_ellipsoid):
    eps = np.finfo(float).eps
    num_pts = obj_mesh.GetNumberOfPoints()
    coords_mat = np.zeros((num_pts, 3))
    for i in range(num_pts):
        pt = obj_mesh.GetPoint(i)
        coords_mat[i, :] = pt
    input_center = np.mean(coords_mat, axis=0)[np.newaxis, :]

    num_pts = standard_ellipsoid.GetNumberOfPoints()
    coords_mat = np.zeros((num_pts, 3))
    for i in range(num_pts):
        pt = standard_ellipsoid.GetPoint(i)
        coords_mat[i, :] = pt
    ell_center = np.mean(coords_mat, axis=0)[np.newaxis, :]

    transVec = input_center[0] - ell_center[0]

    for i in range(num_pts):
        ell_pt = standard_ellipsoid.GetPoint(i)
        pt = list(ell_pt)
        pt = [pt[0] + transVec[0], pt[1] + transVec[1], pt[2] + transVec[2]]
        pt = tuple(pt)
        standard_ellipsoid.GetPoints().SetPoint(i, pt)


    num_pts = standard_ellipsoid.GetNumberOfPoints()
    num_crest_points = 24

    coords_mat = np.zeros((num_pts, 3))
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for i in range(num_pts):
        pt = standard_ellipsoid.GetPoint(i)
        coords_mat[i, :] = pt
        # if i % 5 == 0:
        #     ax.scatter(pt[0], pt[1], pt[2])
    input_center = np.mean(coords_mat, axis=0)[np.newaxis, :]
    centered_coords = coords_mat - input_center

    covariance = np.cov(centered_coords.T) / num_pts
    _, s, vh = np.linalg.svd(covariance)
    rx, ry, rz = 2 * np.sqrt(s * num_pts).T
    #    plt.show()
    #     mass_filter = vtk.vtkMassProperties()
    #     mass_filter.SetInputData(standard_ellipsoid)
    #     mass_filter.Update()
    #     volume = mass_filter.GetVolume()

    #     rx, ry, rz = np.sqrt(s)

    #     ellipsoid_volume = 4 / 3.0 * np.pi * rx * ry * rz
    #     volume_factor = pow(volume/ ellipsoid_volume, 1.0 / 3.0)

    volume_factor = 0.8
    ### notations consistent with wenqi's presentation
    rx *= volume_factor
    ry *= volume_factor
    rz *= volume_factor

    mrx_o = (rx*rx-rz*rz)/rx
    mry_o = (ry*ry-rz*rz)/ry

    ELLIPSE_SCALE = 0.9
    mrb = mry_o * ELLIPSE_SCALE
    mra = mrx_o * ELLIPSE_SCALE

    delta_theta = 2 * np.pi / num_crest_points
    num_steps = 3
    skeletal_pts_x = np.zeros((num_crest_points, num_steps))
    skeletal_pts_y = np.zeros((num_crest_points, num_steps))
    skeletal_pts_z = np.zeros((num_crest_points, num_steps))
    bdry_up_x = np.zeros((num_crest_points, num_steps))
    bdry_up_y = np.zeros((num_crest_points, num_steps))
    bdry_up_z = np.zeros((num_crest_points, num_steps))

    bdry_down_x = np.zeros((num_crest_points, num_steps))
    bdry_down_y = np.zeros((num_crest_points, num_steps))
    bdry_down_z = np.zeros((num_crest_points, num_steps))

    crest_bdry_pts = np.zeros((num_crest_points, 3))
    crest_skeletal_pts = np.zeros((num_crest_points, 3))
    for i in range(num_crest_points):
        theta = np.pi - delta_theta * i
        x = mra * np.cos(theta)
        y = mrb * np.sin(theta)

        mx_ = (mra * mra - mrb * mrb) * np.cos(theta) / mra
        my_ = .0
        dx_ = x - mx_
        dy_ = y - my_

        step_size = 1.0 / float(num_steps-1)

        for j in range(num_steps):
            sp_x = mx_ + step_size * j * dx_
            sp_y = my_ + step_size * j * dy_

            skeletal_pts_x[i, j] = sp_x
            skeletal_pts_y[i, j] = sp_y
            sin_spoke_angle = sp_y * mrx_o
            cos_spoke_angle = sp_x * mry_o

            # normalize to [-1, 1]
            l = np.sqrt(sin_spoke_angle ** 2 + cos_spoke_angle ** 2)
            if l >  eps:
                sin_spoke_angle /= l
                cos_spoke_angle /= l
            cos_phi = l / (mrx_o * mry_o)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            bdry_x = rx * cos_phi * cos_spoke_angle
            bdry_y = ry * cos_phi * sin_spoke_angle
            bdry_z = rz * sin_phi
            bdry_up_x[i, j] = bdry_x
            bdry_up_y[i, j] = bdry_y
            bdry_up_z[i, j] = bdry_z

            bdry_down_x[i, j] = bdry_x
            bdry_down_y[i, j] = bdry_y
            bdry_down_z[i, j] = -bdry_z

            ## if at the boundary of the ellipse, add crest spokes
            if j == num_steps - 1:
                cx = rx * cos_spoke_angle - sp_x
                cy = ry * sin_spoke_angle - sp_y
                cz = 0
                vec_c = np.asarray([cx, cy, cz])
                norm_c = np.linalg.norm(vec_c)
                dir_c = np.asarray([bdry_x - sp_x, bdry_y - sp_y, 0.0])
                dir_c = dir_c / np.linalg.norm(vec_c)

                crest_spoke = norm_c * dir_c
                crest_bdry_x = crest_spoke[0] + sp_x
                crest_bdry_y = crest_spoke[1] + sp_y
                crest_bdry_z = 0.0

                crest_bdry_pts[i] = np.asarray([crest_bdry_x, crest_bdry_y, crest_bdry_z])
                crest_skeletal_pts[i] = np.asarray([sp_x, sp_y, 0.0])
    ### Rotate skeletal/implied boundary points as boundary points of the ellipsoid
    rot_obj = np.flipud(vh.T)
    ## make this rotation matrix same with c++ computation with Eigen3
    # rot_obj[0, :] *= -1
    # rot_obj[-1, :] *= -1

    concate_skeletal_pts = np.concatenate((skeletal_pts_x.flatten()[:, np.newaxis], \
                                        skeletal_pts_y.flatten()[:, np.newaxis], \
                                        skeletal_pts_z.flatten()[:, np.newaxis]), \
                                                axis=1)
    concate_bdry_up_pts = np.concatenate((bdry_up_x.flatten()[:, np.newaxis], \
                                    bdry_up_y.flatten()[:, np.newaxis], \
                                    bdry_up_z.flatten()[:, np.newaxis]), axis=1)
    concate_bdry_down_pts = np.concatenate((bdry_down_x.flatten()[:, np.newaxis], \
                                            bdry_down_y.flatten()[:, np.newaxis], \
                                            bdry_down_z.flatten()[:, np.newaxis]), axis=1)

    second_moment_srep = np.matmul(concate_skeletal_pts.T, concate_skeletal_pts)
    s_srep, v_srep = np.linalg.eig(second_moment_srep)

    rot_srep = v_srep

    rotation = np.matmul(rot_obj, rot_srep)
    rotation = np.flipud(rotation)

    transformed_concate_skeletal_pts = np.matmul(concate_skeletal_pts, rotation) + input_center
    transformed_concate_bdry_up_pts = np.matmul(concate_bdry_up_pts, rotation) + input_center
    transformed_concate_bdry_down_pts = np.matmul(concate_bdry_down_pts, rotation) + input_center
    transformed_crest_bdry_pts = np.matmul(crest_bdry_pts, rotation) + input_center
    transformed_crest_skeletal_pts = np.matmul(crest_skeletal_pts, rotation) + input_center

    ### Convert spokes to visualizable elements
    up_spokes_poly = vtk.vtkPolyData()
    up_spokes_pts = vtk.vtkPoints()
    up_spokes_cells = vtk.vtkCellArray()
    down_spokes_poly = vtk.vtkPolyData()
    down_spokes_pts = vtk.vtkPoints()
    down_spokes_cells = vtk.vtkCellArray()
    crest_spokes_poly = vtk.vtkPolyData()
    crest_spokes_pts = vtk.vtkPoints()
    crest_spokes_cells = vtk.vtkCellArray()

    for i in range(concate_skeletal_pts.shape[0]):
        id_s = up_spokes_pts.InsertNextPoint(transformed_concate_skeletal_pts[i, :])
        id_b = up_spokes_pts.InsertNextPoint(transformed_concate_bdry_up_pts[i, :])

        id_sdwn = down_spokes_pts.InsertNextPoint(transformed_concate_skeletal_pts[i, :])
        id_down = down_spokes_pts.InsertNextPoint(transformed_concate_bdry_down_pts[i, :])

        up_spoke = vtk.vtkLine()
        up_spoke.GetPointIds().SetId(0, id_s)
        up_spoke.GetPointIds().SetId(1, id_b)
        up_spokes_cells.InsertNextCell(up_spoke)

        down_spoke = vtk.vtkLine()
        down_spoke.GetPointIds().SetId(0, id_sdwn)
        down_spoke.GetPointIds().SetId(1, id_down)
        down_spokes_cells.InsertNextCell(down_spoke)


    up_spokes_poly.SetPoints(up_spokes_pts)
    up_spokes_poly.SetLines(up_spokes_cells)
    down_spokes_poly.SetPoints(down_spokes_pts)
    down_spokes_poly.SetLines(down_spokes_cells)

    for i in range(num_crest_points):
        id_crest_s = crest_spokes_pts.InsertNextPoint(transformed_crest_skeletal_pts[i, :])
        id_crest_b = crest_spokes_pts.InsertNextPoint(transformed_crest_bdry_pts[i, :])
        crest_spoke = vtk.vtkLine()
        crest_spoke.GetPointIds().SetId(0, id_crest_s)
        crest_spoke.GetPointIds().SetId(1, id_crest_b)
        crest_spokes_cells.InsertNextCell(crest_spoke)
    crest_spokes_poly.SetPoints(crest_spokes_pts)
    crest_spokes_poly.SetLines(crest_spokes_cells)

    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputData(up_spokes_poly)
    append_filter.AddInputData(down_spokes_poly)
    append_filter.AddInputData(crest_spokes_poly)
    append_filter.Update()


    srep_poly = append_filter.GetOutput()
    input_mesh = standard_ellipsoid

    num_spokes = srep_poly.GetNumberOfCells()
    num_pts = srep_poly.GetNumberOfPoints()
    radii_array = np.zeros(num_spokes)
    dir_array = np.zeros((num_spokes, 3))
    base_array = np.zeros((num_spokes,3))

    ### read the parameters from s-rep
    for i in range(num_spokes):
        id_base_pt = i * 2
        id_bdry_pt = id_base_pt + 1
        base_pt = np.array(srep_poly.GetPoint(id_base_pt))
        bdry_pt = np.array(srep_poly.GetPoint(id_bdry_pt))

        radius = np.linalg.norm(bdry_pt - base_pt)
        direction = (bdry_pt - base_pt) / radius

        radii_array[i] = radius
        dir_array[i, :] = direction
        base_array[i, :] = base_pt
    def obj_func(radii, grad=None):
        """
        Square of signed distance from tips
        of spokes to the input_mesh
        """
        implicit_distance = vtk.vtkImplicitPolyDataDistance()
        implicit_distance.SetInput(input_mesh)
        total_loss = 0
        for i, radius in enumerate(radii):
            direction = dir_array[i, :]
            base_pt   = base_array[i, :]
            bdry_pt   = base_pt + radius * direction

            dist = implicit_distance.FunctionValue(bdry_pt)
            total_loss += dist ** 2
        return total_loss
    # from scipy import optimize as opt
    # minimum = opt.fmin(obj_func, radii_array)

    # minimizer = minimum[0]

    ### optimize the variables (i.e., radii)
    import nlopt
    opt = nlopt.opt(nlopt.LN_NEWUOA, len(radii_array))
    opt.set_min_objective(obj_func)
    opt.set_maxeval(2000)
    minimizer = opt.optimize(radii_array)

    ## update radii of s-rep and return the updated
    num_diff_spokes = 0
    arr_length = vtk.vtkDoubleArray()
    arr_length.SetNumberOfComponents(1)
    arr_length.SetName("spokeLength")

    arr_dirs = vtk.vtkDoubleArray()
    arr_dirs.SetNumberOfComponents(3)
    arr_dirs.SetName("spokeDirection")

    for i in range(num_spokes):
        id_base_pt = i * 2
        id_bdry_pt = id_base_pt + 1
        base_pt = base_array[i, :]
        radius = minimizer[i]
        direction = dir_array[i, :]

        new_bdry_pt = base_pt + radius * direction
        arr_length.InsertNextValue(radius)
        arr_dirs.InsertNextTuple(direction)
        srep_poly.GetPoints().SetPoint(id_bdry_pt, new_bdry_pt)

        if np.abs(np.linalg.norm(new_bdry_pt - base_pt) - radii_array[i]) > eps:
            num_diff_spokes += 1
    #        srep_poly.SetPoint

    srep_poly.GetPointData().AddArray(arr_length)
    srep_poly.GetPointData().AddArray(arr_dirs)
    srep_poly.Modified()
    ell_srep = srep_poly

    target_mesh = obj_mesh
    ell_mesh = standard_ellipsoid

    source_pts = vtk.vtkPoints()
    target_pts = vtk.vtkPoints()
    for i in range(num_pts):
        pt = [0] * 3
        ell_mesh.GetPoint(i, pt)
        source_pts.InsertNextPoint(pt)

        target_mesh.GetPoint(i, pt)
        target_pts.InsertNextPoint(pt)
    source_pts.Modified()
    target_pts.Modified()

    ### Interpolate deformation with thin-plate-spline
    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(source_pts)
    tps.SetTargetLandmarks(target_pts)
    tps.SetBasisToR()
    tps.Modified()

    ### Apply the deformation onto the spokes
    deformed_srep = vtk.vtkPolyData()
    deformed_spokes_ends = vtk.vtkPoints()
    deformed_spoke_lines = vtk.vtkCellArray()
    # refined_srep is a polydata that collects spokes

    for i in range(ell_srep.GetNumberOfCells()):
        base_pt_id = i * 2
        bdry_pt_id = i * 2 + 1
        s_pt = ell_srep.GetPoint(base_pt_id)
        b_pt = ell_srep.GetPoint(bdry_pt_id)

        new_s_pt = tps.TransformPoint(s_pt)
        new_b_pt = tps.TransformPoint(b_pt)

        id0 = deformed_spokes_ends.InsertNextPoint(new_s_pt)
        id1 = deformed_spokes_ends.InsertNextPoint(new_b_pt)

        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        deformed_spoke_lines.InsertNextCell(spoke_line)
    deformed_srep.SetPoints(deformed_spokes_ends)
    deformed_srep.SetLines(deformed_spoke_lines)
    deformed_srep.Modified()
    return deformed_srep
