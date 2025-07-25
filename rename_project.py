#!/usr/bin/env python3
import json
import os
import sys
import re

def update_package_json(new_name):
    with open('package.json', 'r') as f:
        data = json.load(f)
    
    data['name'] = new_name
    
    with open('package.json', 'w') as f:
        json.dump(data, f, indent=2)

def update_package_lock(new_name):
    if not os.path.exists('package-lock.json'):
        return
    
    with open('package-lock.json', 'r') as f:
        data = json.load(f)
    
    # Update the name in package-lock.json
    data['name'] = new_name
    
    # Update any references to the old name in dependencies
    old_name = "next-flask-bp"
    for package in data.get('packages', {}):
        if package.startswith(f'node_modules/{old_name}'):
            new_package = package.replace(old_name, new_name)
            data['packages'][new_package] = data['packages'].pop(package)
    
    with open('package-lock.json', 'w') as f:
        json.dump(data, f, indent=2)

def update_readme(new_name):
    if not os.path.exists('README.md'):
        return
    
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Replace the old name with the new name
    content = content.replace('next-flask-bp', new_name)
    
    with open('README.md', 'w') as f:
        f.write(content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python rename_project.py <new-project-name>")
        sys.exit(1)
    
    new_name = sys.argv[1]
    
    # Validate the new name
    if not re.match(r'^[a-z0-9-]+$', new_name):
        print("Error: Project name should only contain lowercase letters, numbers, and hyphens")
        sys.exit(1)
    
    print(f"Renaming project to: {new_name}")
    
    try:
        update_package_json(new_name)
        print("✓ Updated package.json")
        
        update_package_lock(new_name)
        print("✓ Updated package-lock.json")
        
        update_readme(new_name)
        print("✓ Updated README.md")
        
        print("\nProject renamed successfully!")
        print("\nNext steps:")
        print("1. Run 'npm install' to update dependencies")
        print("2. Update any other files that might contain the old project name")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 