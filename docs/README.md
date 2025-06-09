# Mintlify Starter Kit

### Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mintlify) to preview the documentation changes locally. To install, use the following command

```
npm i -g mintlify
```

Run the following command at the root of your documentation (where docs.json is)

```
mintlify dev
```

**Note:** The `docs.json` file is the core configuration that defines your docs' navigation and layout, making it essential for Mintlify to properly run and preview your site.

### Publishing Changes

Our GitHub App is already installed and seamlessly propagates changes from the OASIS repo to https://docs.oasis.camel-ai.org/. Updates are automatically deployed to production whenever changes are pushed to the main branch.

### Troubleshooting

- Mintlify dev isn't running - Run `mintlify install` it'll re-install dependencies.
- Page loads as a 404 - Make sure you are running in a folder with `docs.json`
