import re
import traceback
import os
from tesseract_robotics.tesseract_common import ResourceLocator, SimpleLocatedResource
#resource locator class using GAZEBO_MODEL_PATH and model:// url
class GazeboModelResourceLocator(ResourceLocator):
    def __init__(self):
        super().__init__()
        model_env_path = os.environ["GAZEBO_MODEL_PATH"]
        self.model_paths = model_env_path.split(os.pathsep)
        assert len(self.model_paths) != 0, "No GAZEBO_MODEL_PATH specified!"
        for p in self.model_paths:
            if os.path.isdir(p):
                break
            assert os.path.isdir(p), "GAZEBO_MODEL_PATH directory does not exist: %s" % p

    def locateResource(self,url):
        try:
            try:
                if os.path.exists(url):
                    return SimpleLocatedResource(url, url, self)
            except:
                pass
            url_match = re.match(r"^model:\/\/(\w+)\/(.+)$",url)
            if (url_match is None):
                print("Invalid Gazebo model resource url %s" % url)
                return None
            model_name = url_match.group(1)
            resource_path = os.path.normpath(url_match.group(2))

            for p in self.model_paths:

                fname = os.path.join(p, model_name, resource_path )
                if not os.path.isfile(fname):
                    continue
                return SimpleLocatedResource(url, fname, self)

            return None
        except:
            traceback.print_exc()
            return None