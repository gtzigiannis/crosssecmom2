# Contributing to Cross-Sectional Momentum System

Thank you for your interest in contributing to the Cross-Sectional Momentum Feature Engineering System!

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](https://github.com/gtzigiannis/crosssecmom2/issues) section
2. If not, create a new issue with:
   - A clear, descriptive title
   - Detailed description of the issue or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed
3. **Test your changes**:
   - Ensure all existing tests pass
   - Add new tests for new functionality
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference relevant issues (e.g., "Fixes #123")
5. **Submit a pull request**:
   - Provide a clear description of the changes
   - Link to any related issues
   - Wait for code review

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/crosssecmom2.git
cd crosssecmom2

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Testing

```bash
# Run tests (when test suite is available)
pytest

# Run with coverage
pytest --cov=.
```

## Areas for Contribution

We welcome contributions in the following areas:

1. **Feature Engineering**:
   - New momentum indicators
   - Alternative volatility measures
   - Additional technical indicators

2. **Alpha Models**:
   - New model implementations
   - Model ensemble methods
   - Machine learning approaches

3. **Portfolio Construction**:
   - Risk management enhancements
   - Transaction cost modeling
   - Execution optimization

4. **Performance Analysis**:
   - Additional metrics
   - Visualization tools
   - Regime analysis

5. **Documentation**:
   - Code examples
   - Tutorials
   - API documentation

6. **Testing**:
   - Unit tests
   - Integration tests
   - Performance benchmarks

## Questions?

Feel free to open an issue for any questions or discussions about contributions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
