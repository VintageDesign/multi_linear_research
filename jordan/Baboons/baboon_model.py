import sys


sys.path.append('../jackson/Libraries')
import JacksonsTSPackage as jts


from ltar import LTAR, LTARI

# read the data
df = pd.read_csv(r"Data\\baboon_tensor_15min.txt", delimiter=",")
