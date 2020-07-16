import yaml, sys
from floodmap.surfaceMapping import LakeMaskProcessor

opspec_file = sys.argv[1]
reproject_inputs = False
with open(opspec_file) as f:
    opspecs = yaml.load(f, Loader=yaml.FullLoader)
    lakeMaskProcessor = LakeMaskProcessor(opspecs)
    lakeMaskProcessor.process_lakes( reproject_inputs, format="tif" )

