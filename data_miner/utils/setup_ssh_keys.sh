#!/bin/bash
# Script to set up SSH key authentication for passwordless sync

echo "Setting up SSH key authentication for passwordless sync..."
echo "=================================================="

# Check if SSH key already exists
if [ -f ~/.ssh/id_rsa.pub ]; then
    echo "✓ SSH key already exists at ~/.ssh/id_rsa.pub"
    echo "Public key content:"
    cat ~/.ssh/id_rsa.pub
    echo ""
else
    echo "Creating new SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    echo "✓ SSH key created"
fi

echo "Next steps:"
echo "1. Copy the public key above"
echo "2. Add it to your remote server's ~/.ssh/authorized_keys file"
echo ""
echo "To copy the key to remote server automatically (if you have password access):"
echo "  ssh-copy-id username@your_server_ip"
echo ""
echo "To test passwordless connection:"
echo "  ssh username@your_server_ip 'echo Connection successful'"
echo ""
echo "Alternative: Manual setup on remote server:"
echo "1. SSH to your remote server"
echo "2. mkdir -p ~/.ssh"
echo "3. echo 'YOUR_PUBLIC_KEY_HERE' >> ~/.ssh/authorized_keys"
echo "4. chmod 700 ~/.ssh"
echo "5. chmod 600 ~/.ssh/authorized_keys"