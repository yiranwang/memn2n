requirejs.config({
    baseUrl: "js",
    paths: {
        "jquery": "lib/jquery/jquery",
        "underscore": "lib/underscore/underscore",
        "backbone": "lib/backbone/backbone"
    },
    shim: {
        "underscore": {
            deps: [],
            exports: "_"
        },
        "backbone": {
            deps: ["jquery", "underscore"],
            exports: "Backbone"
        },
    }
});

define(["app"], function(App) {
});
