import sys

print("Testing Python and Package Installation...")
print(f"Python version: {sys.version}")

packages = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']

for package in packages:
    try:
        if package == 'sklearn':
            import sklearn
            print(f"✓ scikit-learn version: {sklearn.__version__}")
        else:
            mod = __import__(package)
            print(f"✓ {package} version: {mod.__version__}")
    except ImportError as e:
        print(f"✗ {package}: {e}")
    except AttributeError:
        print(f"✓ {package} installed")

print("\nInstallation test complete!")