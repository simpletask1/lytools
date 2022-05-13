import torch
import io


def jit_eg():
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 10

    m = torch.jit.script(MyModule())

    # Save to file
    torch.jit.save(m, 'scriptmodule.pt')
    # This line is equivalent to the previous
    m.save("scriptmodule.pt")

    # Save to io.BytesIO buffer
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)

    # Save with extra files
    extra_files = {'foo.txt': b'bar'}
    torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)

    ############
    import torchvision.models as models
    resnet18_model = models.resnet18()
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    script_model = torch.jit.trace(resnet18_model, input_data)
    script_model.save("resnet18.pt")