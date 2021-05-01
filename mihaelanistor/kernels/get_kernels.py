from kernels.extended_projection_kernels import GrassmannianRBFKernel, GrassmannianProjectionKernel
from kernels.extended_projection_kernels import LinearProjectionKernel, AffineProjectionKernel
from kernels.extended_projection_kernels import LinearScaledProjectionKernel, AffineScaledProjectionKernel
from kernels.extended_projection_kernels import LinearSpherizedProjectionKernel, AffineSpherisedProjectionKernel

def get_callable_kernels():
    return [GrassmannianRBFKernel, GrassmannianProjectionKernel, 
            LinearProjectionKernel, AffineProjectionKernel, 
            LinearScaledProjectionKernel, AffineScaledProjectionKernel, 
            LinearSpherizedProjectionKernel, AffineSpherisedProjectionKernel]