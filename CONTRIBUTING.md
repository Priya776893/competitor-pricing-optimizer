# Contributing to Competitor Pricing Optimizer

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Features

1. Check existing feature requests
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Implementation ideas (if any)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**:
   ```bash
   python train.py  # Ensure models train successfully
   streamlit run app.py  # Test dashboard
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Add: Description of your changes"
   ```

6. **Push and create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

## Testing

Before submitting:
- Run the full pipeline (scrape → preprocess → train → app)
- Ensure no errors or warnings
- Test with sample data

## Questions?

Feel free to open an issue for any questions or clarifications!

