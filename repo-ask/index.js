const { setupExtension, deactivate } = require('./src/extension');

function activate(context) {
	return setupExtension(context);
}

module.exports = {
	activate,
	deactivate
};
