# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Security Considerations

Gallium is designed with security in mind:

### SQL Injection Prevention
- All database queries use parameterized statements
- User input is never interpolated directly into SQL
- LIKE wildcards are properly escaped

### Path Traversal Prevention
- File paths are validated and resolved before use
- Only regular files can be loaded (not directories or special files)

### Input Validation
- Filter parameters are validated against a whitelist
- Invalid filters raise `InvalidFilterError` with clear messages

### Thread Safety
- Optional dependency checks use proper locking
- Database connections use WAL mode for concurrent access

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT open a public issue**
2. Email the maintainer directly at marco.z.difraia@gmail.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You can expect:
- Acknowledgment within 48 hours
- Regular updates on progress
- Credit in the fix announcement (unless you prefer anonymity)

## Security Best Practices for Users

When using Gallium:

1. **Database files**: Store `gallium.db` in a location with appropriate permissions
2. **Image paths**: Be cautious when loading images from untrusted paths
3. **Export files**: Review exported HTML/JSON before sharing publicly
4. **User input**: Validate prompt strings if accepting user input in your application
