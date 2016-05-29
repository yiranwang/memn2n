define(['template/input-template', 'backbone'], function(inputTemplate, Backbone) {

    InputView = Backbone.View.extend({
        initialize: function() {
            this.setElement("#input_container");
            this.render();
        },

        render: function() {
            var template = _.template( inputTemplate);
            this.$el.html(template);
        }
    });

    return InputView;
});
