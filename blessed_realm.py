from realm import Realm
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext


class BlessedRealm(Realm, CUDAEnvironmentContext):
    pass