# 🔒 Security Setup Guide

This guide explains how to properly set up the Mirqab project with API keys and credentials.

## ⚠️ IMPORTANT: Never Commit Sensitive Data

The following files should **NEVER** be committed to git:
- `.env` files (contains API keys and secrets)
- Firebase service account JSON files (`*-firebase-adminsdk-*.json`)
- Model files (`*.pth` - they're too large and should be stored separately)
- Any file containing API keys, passwords, or credentials

## 📋 Setup Steps

### 1. Copy the Environment Template

```bash
cp .env_template .env
```

### 2. Get Your API Keys

#### OpenAI API Key (Required)
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key to your `.env` file

#### Firebase Configuration (Required)
1. Go to Firebase Console: https://console.firebase.google.com
2. Select your project (or create one)
3. Go to Project Settings > Service Accounts
4. Click "Generate New Private Key"
5. Save the JSON file to `backend/` directory
6. Update `FIREBASE_CREDENTIALS_PATH` in `.env` to point to this file

#### PromptLayer API Key (Optional - for RAG evaluation)
1. Go to https://promptlayer.com
2. Sign up and get your API key
3. Add to `.env` file

### 3. Update Your `.env` File

Edit `.env` and fill in your actual values:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=backend/your-actual-firebase-file.json

# PromptLayer (optional)
PROMPTLAYER_API_KEY=pl_your-actual-promptlayer-key-here

# Raspberry Pi Authentication
MIRQAB_API_KEY=change-this-to-a-secure-random-key
```

### 4. Download Model Files

The model file `best_deeplabv3_camouflage.pth` is too large for git.

- Store it separately (e.g., Google Drive, Dropbox, or LFS)
- Place it in the project root directory
- It's already in `.gitignore` so it won't be committed

## 🔐 Security Best Practices

### For Developers

1. **Never hardcode API keys** in source code
2. **Always use environment variables** for secrets
3. **Check `.gitignore`** before committing
4. **Review commits** to ensure no secrets are included
5. **Rotate keys** if accidentally exposed

### For Production

1. Use separate `.env` files for different environments
2. Set proper file permissions: `chmod 600 .env`
3. Use secret management services (AWS Secrets Manager, Azure Key Vault, etc.)
4. Enable environment variable injection in deployment platforms
5. Monitor for API key usage anomalies

## 🚫 What's Already Protected

The `.gitignore` file is configured to exclude:

```
# Environment files
.env
.env.local
.env.production
*.env (except .env_template and .env.example)

# API Keys
*api_key*
*secret*

# Firebase credentials
*-firebase-adminsdk-*.json

# Model files
*.pth
*.pkl
```

## ✅ Verification Checklist

Before committing code, verify:

- [ ] No `.env` files are staged
- [ ] No `*-firebase-adminsdk-*.json` files are staged
- [ ] No hardcoded API keys in code
- [ ] All API keys are loaded from environment variables
- [ ] `.gitignore` is properly configured
- [ ] Model files are not staged

## 🆘 If You Accidentally Committed Secrets

1. **Immediately rotate the exposed credentials**
2. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch path/to/sensitive/file" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (⚠️ dangerous - coordinate with team):
   ```bash
   git push origin --force --all
   ```
4. **Report to security team** if in production

## 📝 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4 Vision |
| `FIREBASE_CREDENTIALS_PATH` | Yes | Path to Firebase service account JSON |
| `PROMPTLAYER_API_KEY` | No | PromptLayer API key for RAG evaluation |
| `MIRQAB_API_KEY` | Yes | API key for Raspberry Pi authentication |
| `FIREBASE_STORAGE_BUCKET` | No | Firebase storage bucket name |

## 🔗 Useful Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Firebase Admin SDK Setup](https://firebase.google.com/docs/admin/setup)
- [Environment Variables Best Practices](https://12factor.net/config)
- [Git Secrets Prevention](https://git-secret.io/)

---

**Remember**: Security is everyone's responsibility. When in doubt, ask before committing!
