import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("UI Error Caught by ErrorBoundary:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          maxWidth: '800px',
          margin: '2rem auto',
          background: '#1e1e2e',
          color: '#f38ba8',
          borderRadius: '12px',
          border: '1px solid #f38ba8',
          fontFamily: 'monospace'
        }}>
          <h2>⚠️ Something went wrong in the UI</h2>
          <p style={{ color: '#cdd6f4', marginTop: '1rem' }}>
            <strong>Error:</strong> {this.state.error?.toString()}
          </p>
          {this.state.errorInfo && (
            <details style={{ marginTop: '1rem', whiteSpace: 'pre-wrap', color: '#a6adc8' }}>
              <summary style={{ cursor: 'pointer', color: '#89b4fa' }}>Stack Trace</summary>
              {this.state.errorInfo.componentStack}
            </details>
          )}
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '1.5rem',
              padding: '0.6rem 1.2rem',
              background: '#89b4fa',
              color: '#11111b',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
