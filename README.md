Before running this program for IBM quantum computers (as opposed to the local simulator), you will need to set up the IBM Qiskit Runtime. First, create a file named config.ini, which will contain your IBM credentials. The format is as follows:

```ini
[IBM]
token = <YOUR_IBM_TOKEN>
```

This program will read the config.ini file for your API token. Now, run the following code: 

```python
import configparser
from qiskit_ibm_runtime import QiskitRuntimeService

with open("config.ini", "r") as cf:
    cp = configparser.ConfigParser()
    cp.read_file(cf)
    api_token = cp.get("IBM", "token")
QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, overwrite=True)
```