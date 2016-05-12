defmodule ElixirCudaExample do

  @node :"c1@NEWBORN" # TODO: change this to your longname

  @doc """
  Returns x + 1
  """
  def foo(x), do:
    call_cnode {:foo, x}

  @doc """
  Returns y * 2
  """
  def bar(y), do:
    call_cnode {:bar, y}

  #######################

  defp call_cnode(msg) do
    send {:any, @node}, {:call, self, msg}
    receive do
      {:cnode, result} -> result
    end
  end

end
